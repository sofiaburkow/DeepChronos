### Instruction for network log processing:

- Download the AIT-LDSv2.0 from https://zenodo.org/record/5789064

- All the servers (webserver, inet-firewall, vpn, ...) have logs, usually placed in the `.\<company name>\gather\<server name>\logs\suricata` folder (!! Path to attacker logs: `.\<company name>\gather\attacker_0\logs\ait.aecid.attacker.wpdiscuz\traffic.pcap`)

- Rename them to standard .pcap format (the current name includes the timestamp) - choose meaningful names identifying the server. Note that there may be more than one capture for a single server

    - e.g.: `log.pcap.1642159278 -> log_cloud_share<N>.pcap`
    
- Merge the pcaps captured on the same probe (webserver, inet-firewall, vpn, ...) by using the command `mergecap -a <filenames> -w <final filename>`
    - e.g.: `mergecap -a log_cloud_share1.pcap log_cloud_share2.pcap ... log_cloud_shareN.pcap -w log_cloud_share.pcap`
		
[- Delete the original files that have already been merged (e.g.: `rm -r log_cloud_share1.pcap ... log_cloud_shareN.pcap`)]

- Run **tstat**(*) on the merged files to obtain enhanced netflows with the command `tstat <filename>`:
    - e.g.: `tstat log_cloud_share.pcap`
		-> this will output different log types in different folders, we are interested only in **tcp_complete**, **tcp_nocomplete** and **udp_complete**
        
- Run the python notebook **1_format_dataset_info.ipynb** (update the relative path to the tstat logs where needed), this script will generate the table *simulation_list.csv*, containing all the instructions needed for the labelling and filtering of the flows

- Run the python notebook **2_label_logs.ipynb** (update the company name variable and the relative path to the tstat logs/simulation_list.csv where needed), this script will output the final labelled logs.

(*) Check out http://tstat.polito.it/measure.shtml for info on installation and log files interpretation

Cite Tstat paper when using the tool: Trevisan, Martino, et al. "Traffic analysis with off-the-shelf hardware: Challenges and lessons learned." IEEE Communications Magazine 55.3 (2017): 163-169.