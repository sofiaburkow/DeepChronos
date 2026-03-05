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