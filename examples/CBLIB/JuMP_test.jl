
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
        # # easy
        # "bss1",
        # "demb782",
        # "bss2",
        # "demb781",
        # "gptest",
        # "rijc781",
        # "synthes2",
        # "rijc784",
        # "rijc785",
        # "rijc783",
        # "syn10m",
        # "syn05h",
        # "beck751",
        # "beck752",
        # "beck753",
        # "fiac81b",
        # "syn05m02m",
        # "fang88",
        # "syn10h",
        # "syn05m02h",
        # "rijc787",
        # "syn15h",
        # "syn10m02h",
        # # hard
        # "synthes1",
        # "rijc786",
        # "syn05m",
        # "rijc782",
        # "batchdes",
        # "synthes3",
        # "batch",
        # "syn15m",
        # "demb761",
        # "demb762",
        # "demb763",
        # "syn20m",
        # "syn05m03m",
        # "fiac81a",
        # "syn30m",
        # "syn10m02m",
        # "ravem",
        # "syn05m04m",
        # "enpro56",
        # "syn05m03h",
        # "syn20h",
        # "syn40m",
        # "rsyn0805m",
        # "enpro48",
        # "syn15m02m",
        # "syn10m03m",
        # "rsyn0810m",
        # "syn05m04h",
        # "rsyn0815m",
        # "syn30h",
        # "rsyn0820m",
        # "syn20m02m",
        # exp and orthant
        "LogExpCR-n100-m1200",
        "LogExpCR-n100-m1600",
        "LogExpCR-n100-m400",
        "LogExpCR-n100-m800",
        "LogExpCR-n20-m1200",
        "LogExpCR-n20-m1600",
        "LogExpCR-n20-m400",
        "LogExpCR-n20-m800",
        "LogExpCR-n500-m1200",
        "LogExpCR-n500-m1600",
        "LogExpCR-n500-m400",
        "LogExpCR-n500-m800",
        "batch",
        "batchdes",
        "batchs101006m",
        "batchs121208m",
        "batchs151208m",
        "batchs201210m",
        "beck751",
        "beck752",
        "beck753",
        "bss1",
        "bss2",
        "car",
        "demb761",
        "demb762",
        "demb763",
        "demb781",
        "demb782",
        "enpro48",
        "enpro56",
        "fang88",
        "fiac81a",
        "fiac81b",
        "gams01",
        "gp_dave_1",
        "gp_dave_2",
        "gp_dave_3",
        "gptest",
        "isil01",
        "jha88",
        "mra01",
        "ravem",
        "rijc781",
        "rijc782",
        "rijc783",
        "rijc784",
        "rijc785",
        "rijc786",
        "rijc787",
        "rsyn0805h",
        "rsyn0805m",
        "rsyn0805m02h",
        "rsyn0805m02m",
        "rsyn0805m03h",
        "rsyn0805m03m",
        "rsyn0805m04h",
        "rsyn0805m04m",
        "rsyn0810h",
        "rsyn0810m",
        "rsyn0810m02h",
        "rsyn0810m02m",
        "rsyn0810m03h",
        "rsyn0810m03m",
        "rsyn0810m04h",
        "rsyn0810m04m",
        "rsyn0815h",
        "rsyn0815m",
        "rsyn0815m02h",
        "rsyn0815m02m",
        "rsyn0815m03h",
        "rsyn0815m03m",
        "rsyn0815m04h",
        "rsyn0815m04m",
        "rsyn0820h",
        "rsyn0820m",
        "rsyn0820m02h",
        "rsyn0820m02m",
        "rsyn0820m03h",
        "rsyn0820m03m",
        "rsyn0820m04h",
        "rsyn0820m04m",
        "rsyn0830h",
        "rsyn0830m",
        "rsyn0830m02h",
        "rsyn0830m02m",
        "rsyn0830m03h",
        "rsyn0830m03m",
        "rsyn0830m04h",
        "rsyn0830m04m",
        "rsyn0840h",
        "rsyn0840m",
        "rsyn0840m02h",
        "rsyn0840m02m",
        "rsyn0840m03h",
        "rsyn0840m03m",
        "rsyn0840m04h",
        "rsyn0840m04m",
        "syn05h",
        "syn05m",
        "syn05m02h",
        "syn05m02m",
        "syn05m03h",
        "syn05m03m",
        "syn05m04h",
        "syn05m04m",
        "syn10h",
        "syn10m",
        "syn10m02h",
        "syn10m02m",
        "syn10m03h",
        "syn10m03m",
        "syn10m04h",
        "syn10m04m",
        "syn15h",
        "syn15m",
        "syn15m02h",
        "syn15m02m",
        "syn15m03h",
        "syn15m03m",
        "syn15m04h",
        "syn15m04m",
        "syn20h",
        "syn20m",
        "syn20m02h",
        "syn20m02m",
        "syn20m03h",
        "syn20m03m",
        "syn20m04h",
        "syn20m04m",
        "syn30h",
        "syn30m",
        "syn30m02h",
        "syn30m02m",
        "syn30m03h",
        "syn30m03m",
        "syn30m04h",
        "syn30m04m",
        "syn40h",
        "syn40m",
        "syn40m02h",
        "syn40m02m",
        "syn40m03h",
        "syn40m03m",
        "syn40m04h",
        "syn40m04m",
        "synthes1",
        "synthes2",
        "synthes3",
        "varun",
        # exp and epinormecul
        # "ex1223",
        # "ex1223b",
        # "gams01",
        # "port_12_9_3_a_1",
        # "port_12_9_3_a_2",
        # "port_12_9_3_b_1",
        # "port_12_9_3_b_2",
        # "port_12_9_3_c_1",
        # "port_12_9_3_c_2",
        # "port_12_9_3_d_1",
        # "port_12_9_3_d_2",
        # "port_16_12_4_a_1",
        # "port_16_12_4_a_2",
        # "port_16_12_4_b_1",
        # "port_16_12_4_b_2",
        # "port_16_12_4_c_1",
        # "port_16_12_4_c_2",
        # "port_16_12_4_d_1",
        # "port_16_12_4_d_2",
        # "port_20_15_5_a_1",
        # "port_20_15_5_a_2",
        # "port_20_15_5_b_1",
        # "port_20_15_5_b_2",
        # "port_20_15_5_c_1",
        # "port_20_15_5_c_2",
        # "port_20_15_5_d_1",
        # "port_20_15_5_d_2",
        ]
end

relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [(("expdesign_D_8_4",), nothing, relaxed_tols)]
insts["fast"] = [((inst,), nothing, relaxed_tols) for inst in cblib_diverse]
insts["various"] = insts["fast"]
return (CBLIBJuMP, insts)
