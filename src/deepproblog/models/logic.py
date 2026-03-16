from problog.extern import problog_export, problog_export_nondet


@problog_export_nondet('+term')
def homenet_ip(VictimIP):
    """
    Boolean-style predicate: succeeds when VictimIP is in the homenet.
    """
    VictimIP = str(VictimIP)
    is_homenet = VictimIP.startswith("172.16.")
    # print(f"is_homenet({VictimIP}) -> {is_homenet}")
    return [()] if is_homenet else []

@problog_export_nondet('+term')
def external_ip(VictimIP):
    """
    Boolean-style predicate: succeeds when VictimIP is an external IP.
    """
    VictimIP = str(VictimIP)
    is_external = not VictimIP.startswith("172.16.")
    # print(f"is_external({VictimIP}) -> {is_external}")
    return [()] if is_external else []