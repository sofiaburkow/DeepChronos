from problog.extern import problog_export, problog_export_nondet


@problog_export_nondet('+term')
def is_homenet(VictimIP):
    """
    Boolean-style predicate: succeeds when VictimIP is in the homenet.
    """
    VictimIP = str(VictimIP)
    is_homenet = VictimIP.startswith("172.16.")
    # print(f"is_homenet({VictimIP}) -> {is_homenet}")
    return [()] if is_homenet else []

@problog_export('+term', '+term')
def is_sadmin_port(Port):
    """
    Boolean-style predicate: succeeds when Port is a known sadmin port.
    """
    port = str(Port)
    is_sadmin_port = port in {"514", "23", "1022"}
    return [()] if is_sadmin_port else []