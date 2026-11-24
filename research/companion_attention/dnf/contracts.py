"""Independent shape contracts from the paper and recovered hand interfaces."""

CONTRACTS = {
    "backward": {
        "inputs": {"Q": ("B","N","D"),"K":("B","N","D"),"V":("B","N","E"),
                   "A":("B","N","N"),"U":("B","N","E")},
        "outputs":{"GV":("B","N","E"),"GQ":("B","N","D"),"GK":("B","N","D")}},
    "fused": {
        "inputs":{"Q":("B","N","D"),"K":("B","N","D"),"V":("B","N","E"),"U":("B","N","E")},
        "outputs":{"O":("B","N","E"),"GV":("B","N","E"),"GQ":("B","N","D"),"GK":("B","N","D")}},
    "decode": {
        "inputs":{"Q":("H","D"),"K":("G","N","D"),"V":("G","N","E")},
        "outputs":{"O":("H","E")}},
    "rmsnorm": {
        "inputs":{"X":("B","N","D"),"U":("B","N","D"),"W":("D",)},
        "outputs":{"Z":("B","N","D"),"GX":("B","N","D"),"GW":("D",)}},
    "ffn": {
        "inputs":{"X":("T","D"),"U":("T","D"),"Wg":("D","F"),"Wu":("D","F"),"Wd":("F","D")},
        "outputs":{"Y":("T","D"),"GX":("T","D"),"GWg":("D","F"),"GWu":("D","F"),"GWd":("F","D")}},
    "append": {
        "inputs":{"key":("G","D"),"value":("G","E")},
        "outputs":{}},
    "block": {
        "inputs":{"X":("T","D"),"scale1":("D",),"scale2":("D",),
                  "Wq":("D","D"),"Wk":("D","D"),"Wv":("D","D"),
                  "Wg":("D","F"),"Wu":("D","F"),"Wd":("F","D"),"GY":("T","D")},
        "outputs":{"Y":("T","D"),"GX":("T","D"),"Gscale1":("D",),"Gscale2":("D",),
                   "GWq":("D","D"),"GWk":("D","D"),"GWv":("D","D"),
                   "GWg":("D","F"),"GWu":("D","F"),"GWd":("F","D")}},
}


def validate(name, kernel):
    shapes,inputs,outputs,_=kernel
    expected=CONTRACTS[name]
    if set(inputs)!=set(expected["inputs"]) or set(outputs)!=set(expected["outputs"]):
        raise AssertionError(f"Observable interface changed: {name}")
    for category in ("inputs","outputs"):
        for array,shape in expected[category].items():
            if tuple(shapes[array])!=shape:
                raise AssertionError(f"Shape mismatch: {name}.{array}")
    if name=="append":
        assert tuple(shapes["K"])==("G","C","D")
        assert tuple(shapes["V"])==("G","C","E")
    return expected
