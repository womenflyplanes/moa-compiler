"""Lower validated scalar ASTs through the checked Fortran/MATLAB schedules."""
import re
import textwrap
from functools import wraps
from semantic_ir import validate, render


def wrap_generator(name, fn):
    family='matmul' if 'matmul' in name else 'island' if 'island' in name else 'rbgs' if 'rbgs' in name else 'jacobi'
    mode='mpi' if name.startswith('gen_mpi') else 'openmp' if name.startswith('gen_openmp') else 'gpu' if name.startswith('gen_gpu') else 'matlab' if name.startswith('gen_matlab') else 'serial'
    @wraps(fn)
    def emit(data):
        validate(data)
        from moa_stages import lift
        plan=lift(data,'openacc' if mode=='gpu' else mode)
        if plan['onf']['arithmetic_tree']!=data['update']:raise ValueError('ONF arithmetic mismatch')
        if data['family']!=family:raise ValueError('Generator family and DNF disagree')
        snapshot=family=='island' and data['state_semantics']=='snapshot_islands'
        if family=='island' and mode in ('mpi','gpu','openmp') and not snapshot:
            raise ValueError('Parallel island mapping requires explicit snapshot_islands transformation; it changes finite-sweep semantics')
        if snapshot and mode in ('serial','matlab'):
            raise ValueError('Snapshot serial/MATLAB schedule is not implemented; cannot silently change state reads')
        text=fn(data['dnf'])
        lang='matlab' if mode=='matlab' else 'fortran'
        rhs=render(data['update'],lang,island=snapshot,mpi=mode=='mpi')
        # Replace the entire numerical RHS, preserving the parsed arithmetic tree.
        count=0;lines=[]
        for line in text.splitlines():
            if family=='matmul':match=re.match(r'(\s*C\(i,j\)\s*=)\s*C\(i,j\)\s*\+.*',line,re.I)
            else:match=re.match(r'(\s*u(?:_new)?\(i,\s*(?:j|j_local)\)\s*=).*',line,re.I)
            if match:
                line=match[1]+' '+rhs+(';' if mode=='matlab' else '')
                count+=1
            lines.append(line)
        if count!=1:raise ValueError('Expected exactly one lowered numerical update')
        text='\n'.join(lines)+'\n'
        # Scalar shape/iteration parameters are semantic input, not driver constants.
        for key,value in data['parameters'].items():
            if key in ('nix','niy'):continue
            if mode=='matlab':
                text=re.sub(r'\b'+key+r'\s*=\s*(?:64|100|16)(?=[;,])',key+' = '+str(value),text)
            else:
                text=re.sub(r'(?i)(\b'+key+r'\s*=\s*)(?:64|100|16)(?=[,\s\n])',lambda m:m[1]+str(value),text)
            if key=='h':text=re.sub(r'h = 1\.0 / 63\.0', 'h = '+repr(value), text)
        # MATLAB calls may not override a compiled contract without regenerating IR.
        if mode=='matlab':
            args=['n'] if family=='matmul' else ['nx','ny','niter']
            check=' || '.join(k+' ~= '+str(data['parameters'][k]) for k in args)
            lines=text.splitlines();at=next(i for i,x in enumerate(lines) if 'if nargin' in x)
            lines.insert(at+1, "  if "+check+", error('Shape/iteration contract changed: regenerate semantic IR'); end")
            at=next(i for i,x in enumerate(lines) if x.lstrip().startswith('for '))
            lines.insert(at, '  A=single(A); B=single(B); C=single(C);' if family=='matmul' else '  u=single(u); f=single(f); h=single(h);'+(' u_new=single(u_new);' if family=='jacobi' else ''))
            text='\n'.join(lines)+'\n'
        # Fix remainder coverage; no silent dropped rows for non-divisible process counts.
        if mode=='mpi':
            extent='n' if family=='matmul' else 'ny'
            guard=f'''\n  if (nprocs > {extent}) then
    if (rank == 0) print *, 'Unsupported: more MPI ranks than owned rows'
    call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if'''
            if family=='island':
                guard+='''\n  if (mod(ny,nprocs) /= 0 .or. mod(ny/nprocs,isy) /= 0) then
    if (rank == 0) print *, 'Unsupported: MPI slabs must align with island rows'
    call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
  end if'''
            text=text.replace('  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)','  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)'+guard)
            if family=='jacobi':
                text=text.replace('  j0_global = rank*ny_local + 1','  if (rank < mod(ny,nprocs)) ny_local = ny_local + 1\n  j0_global = rank*(ny/nprocs) + min(rank,mod(ny,nprocs)) + 1')
                text=text.replace('  j1_global = (rank+1)*ny_local','  j1_global = j0_global + ny_local - 1')
            if family=='matmul':
                text=text.replace('  i0_global = rank*n_local + 1','  if (rank < mod(n,nprocs)) n_local = n_local + 1\n  i0_global = rank*(n/nprocs) + min(rank,mod(n,nprocs)) + 1')
        if mode=='gpu' and snapshot:
            text=text.replace('        do j = j0, j1','        !$ACC LOOP SEQ\n        do j = j0, j1')
            text=text.replace('          do i = i0, i1','          !$ACC LOOP SEQ\n          do i = i0, i1')
        # tmp is overwritten on device before read; avoid copying undefined host values.
        text=text.replace('COPY(u, u_new, f, tmp)', 'COPY(u, u_new, f) CREATE(tmp)')
        # Fortran standard line limits, including directive continuation.
        if mode!='matlab':
            result=[]
            for line in text.splitlines():
                if len(line)<=120 or (line.lstrip().startswith('!') and not line.lstrip().lower().startswith(('!$omp','!$acc'))):
                    result.append(line);continue
                prefix='!$OMP& ' if line.lstrip().lower().startswith('!$omp') else '!$ACC& ' if line.lstrip().lower().startswith('!$acc') else '    & '
                while len(line)>120:
                    cut=max(line.rfind(',',0,110),line.rfind(' ',0,110))
                    if cut<15:raise ValueError('Cannot safely wrap generated line')
                    if line[cut]==',':cut+=1
                    result.append(line[:cut]+' &');line=prefix+line[cut:].lstrip()
                result.append(line)
            text='\n'.join(result)+'\n'
        text=text.replace('islands are independent in this sweep - this loop is the parallel loop','Lexicographic island order is part of this DNF state contract')
        text=text.replace('【entity-MATLAB¦canonical_name=MATLAB】','MATLAB')
        return text
    return emit
