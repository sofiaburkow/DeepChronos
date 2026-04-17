nn(net1, [X], Z, [benign, attack]) :: msa(X, Z).

% Traffic pattern rules

home_orig(1).
home_resp(1).
ext_orig(0).
ext_resp(0).

icmp(1).
tcp(6).
udp(17).

icmp_req(SrcO,DstO,Proto) :-
    ext_orig(SrcO),
    home_resp(DstO),
    icmp(Proto).

icmp_resp(SrcO,DstO,Proto) :-
    home_orig(SrcO),
    ext_resp(DstO),
    icmp(Proto).

udp_req(SrcO,DstO,Proto) :-
    ext_orig(SrcO),
    home_resp(DstO),
    udp(Proto).

tcp_req(SrcO,DstO,Proto) :-
    ext_orig(SrcO),
    home_resp(DstO),
    tcp(Proto).

% Vulnerability / expert knowledge

sadmind_port(111).

telnet(23).
rsh(514).

% privileged_port(P) :- P < 1024.
privileged_port(1020).
privileged_port(1021).
privileged_port(1022).
privileged_port(1023).

% DARPA MSA phase rules

phase_rule(SrcO,DstO,DPort,Proto,DDoSSig,phase1) :-
    icmp_req(SrcO,DstO,Proto).

phase_rule(SrcO,DstO,DPort,Proto,DDoSSig,phase2) :-
    udp_req(SrcO,DstO,Proto),
    sadmind_port(DPort).

t(0.7)::phase_rule(SrcO,DstO,DPort,Proto,DDoSSig,phase2) :-
    udp_req(SrcO,DstO,Proto),
    \+ sadmind_port(DPort).

phase_rule(SrcO,DstO,DPort,Proto,DDoSSig,phase2) :-
    icmp_resp(SrcO,DstO,Proto).

phase_rule(SrcO,DstO,DPort,Proto,DDoSSig,phase3) :-
    udp_req(SrcO,DstO,Proto).

phase_rule(SrcO,DstO,DPort,Proto,DDoSSig,phase4) :-
    tcp_req(SrcO,DstO,Proto),
    telnet(DPort).

phase_rule(SrcO,DstO,DPort,Proto,DDoSSig,phase4) :-
    (rsh(DPort); privileged_port(DPort)),
    tcp(Proto).

ddos_sig(1).

t(1.0)::phase_rule(SrcO,DstO,DPort,Proto,DDoSSig,phase5) :-
    tcp(Proto),
    ddos_sig(DDoSSig).

% Valid phase progressions

phase(phase1).
phase(phase2).
phase(phase3).
phase(phase4).
phase(phase5).

t(0.7)::valid_phase(0,0,0,0,phase1).
t(1.0)::valid_phase(1,0,0,0,phase1).

t(0.7)::valid_phase(1,0,0,0,phase2).
t(1.0)::valid_phase(1,1,0,0,phase2).

t(0.7)::valid_phase(1,1,0,0,phase3).
t(1.0)::valid_phase(1,1,1,0,phase3).

t(0.7)::valid_phase(1,1,1,0,phase4).
t(1.0)::valid_phase(1,1,1,1,phase4).

is_compromised(1).

% Multi-step attack logic

multi_step(P1,P2,P3,P4,Compromised,X,SrcO,DstO,DPort,Proto,DDoSSig,Phase) :-
    valid_phase(P1,P2,P3,P4,Phase),
    phase_rule(SrcO,DstO,DPort,Proto,DDoSSig,Phase),
    msa(X,attack).

multi_step(P1,P2,P3,P4,Compromised,X,SrcO,DstO,DPort,Proto,DDoSSig,phase5) :-
    is_compromised(Compromised),
    phase_rule(SrcO,DstO,DPort,Proto,DDoSSig,phase5),
    msa(X,attack).

multi_step(P1,P2,P3,P4,Compromised,X,SrcO,DstO,DPort,Proto,DDoSSig,benign) :-
    \+ (
        phase(Phase),
        multi_step(P1,P2,P3,P4,Compromised,X,SrcO,DstO,DPort,Proto,DDoSSig,Phase)
    ).