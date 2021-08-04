
# directory of CBLIB files
cblib_dir = "C:\\Users\\leaka\\cblib\\cblib.zib.de\\download\\all"
if !isdir(cblib_dir)
    @warn("CBLIB download folder not found")
    cblib_dir = joinpath(@__DIR__, "cblib_data")
    cblib_diverse = String[]
else
    # cblib_diverse = [
    #     # "expdesign_D_8_4", # psd, exp
    #     # "port_12_9_3_a_1", # psd, soc, exp
    #     # "tls4", # soc
    #     # "ck_n25_m10_o1_1", # rsoc
    #     "rsyn0805h", # exp
    #     # "2x3_3bars", # psd
    #     # "HMCR-n20-m400", # power
    #     # "classical_20_0", # soc, orthant
    #     # "achtziger_stolpe06-6.1flowc", # rsoc, orthant
    #     "LogExpCR-n100-m400", # exp, orthant
    #     ]
    cblib_diverse = [
        # easy
        "bss1",
        "demb782",
        "bss2",
        "demb781",
        "gptest",
        "rijc781",
        "synthes2",
        "rijc784",
        "rijc785",
        "rijc783",
        "syn10m",
        "syn05h",
        "beck751",
        "beck752",
        "beck753",
        "fiac81b",
        "syn05m02m",
        "fang88",
        "syn10h",
        "syn05m02h",
        "rijc787",
        "syn15h",
        "syn10m02h",
        # hard
        "synthes1",
        "rijc786",
        "syn05m",
        "rijc782",
        "batchdes",
        "synthes3",
        "batch",
        "syn15m",
        "demb761",
        "demb762",
        "demb763",
        "syn20m",
        "syn05m03m",
        "fiac81a",
        "syn30m",
        "syn10m02m",
        "ravem",
        "syn05m04m",
        "enpro56",
        "syn05m03h",
        "syn20h",
        "syn40m",
        "rsyn0805m",
        "enpro48",
        "syn15m02m",
        "syn10m03m",
        "rsyn0810m",
        "syn05m04h",
        "rsyn0815m",
        "syn30h",
        "rsyn0820m",
        "syn20m02m",
        ]
end

relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [(("expdesign_D_8_4",), nothing, relaxed_tols)]
insts["fast"] = [((inst,), nothing, relaxed_tols) for inst in cblib_diverse]
insts["various"] = insts["fast"]
return (CBLIBJuMP, insts)
