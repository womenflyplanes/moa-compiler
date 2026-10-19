#!/usr/bin/env python3
"""OpenMP extension of the validated RC1 attention C emitter."""

from pathlib import Path
from moa_codegen_v1 import Loop, lift_loop, emit_c


def emit_openmp(query, key):
    source = emit_c(query, key)

    # Allocate one N-element score buffer per query.
    # Allocation and failure handling occur outside the OpenMP region.
    allocation = (
        " double *workspace=(double*)malloc("
        "(size_t)B*(size_t)N*(size_t)N*sizeof(double));\n"
        " if(!workspace)return -2;\n"
    )

    marker = " for(int b=0;b<B;++b){\n"
    if source.count(marker) != 1:
        raise RuntimeError("Expected batch loop not found")

    source = source.replace(marker, allocation + marker, 1)

    old_buffer = (
        "    double *s=(double*)malloc((size_t)N*sizeof(double));\n"
        "    if(!s)return -2;\n"
    )
    new_buffer = (
        "    double *s=workspace+((size_t)b*N+i)*N;\n"
    )

    if source.count(old_buffer) != 1:
        raise RuntimeError("Expected score-buffer allocation not found")

    source = source.replace(old_buffer, new_buffer, 1)

    if source.count("    free(s);\n") != 1:
        raise RuntimeError("Expected score-buffer cleanup not found")

    source = source.replace("    free(s);\n", "", 1)

    target = (
        f"  for(int {query.outer}=0;"
        f"{query.outer}<{query.outer_extent};"
        f"++{query.outer}){{"
    )

    if source.count(target) != 1:
        raise RuntimeError("Expected query loop not found")

    source = source.replace(
        target,
        "  #pragma omp parallel for schedule(static)\n" + target,
        1
    )

    ending = " }\n return 0;\n}\n"
    if source.count(ending) != 1:
        raise RuntimeError("Expected function ending not found")

    source = source.replace(
        ending,
        " }\n free(workspace);\n return 0;\n}\n",
        1
    )

    return source


if __name__ == "__main__":
    query = lift_loop(Loop("i", 5), 2)
    key = lift_loop(Loop("j", 5), 3)

    source = emit_openmp(query, key)

    output = Path(
        "research/attention_rc1/moa_generated_attention_openmp.c"
    )
    output.write_text(source)

    print("Generated OpenMP attention kernel:", output)
