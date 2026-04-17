


def analyze_phase(df, print_details=True):

    src_ips = df["src_ip"].value_counts()
    dst_ips = df["dst_ip"].value_counts()
    src_ports = df["sport"].value_counts()
    dst_ports = df["dport"].value_counts()

    if print_details:
        print("Total Flows:", len(df))

        print("\n --- IP distribution ---")
        print(f"\nSource IPs ({len(src_ips)}):")
        print(src_ips)
        print(f"\nDestination IPs ({len(dst_ips)}):")
        print(dst_ips)

        print("\n --- Port distribution ---")
        print(f"Source Ports ({len(src_ports)}):")
        print(src_ports)
        print(f"\nDestination Ports ({len(dst_ports)}):")
        print(dst_ports)

    return src_ips, dst_ips, src_ports, dst_ports


def analyze_origin_destination(df, phase_name):
    origins = df["local_orig"].value_counts()
    print(f"{phase_name} Origin Distribution:")
    print(origins)

    destinations = df["local_resp"].value_counts()
    print(f"{phase_name} Destination Distribution:")
    print(destinations)

    