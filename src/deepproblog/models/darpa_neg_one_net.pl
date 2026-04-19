nn(net1, [X], Z, [benign, attack]) :: msa(X, Z).

% Valid phase progressions

phase(phase1).
phase(phase2).
phase(phase3).
phase(phase4).
phase(phase5).

valid_phase_progression(P1,P2,P3,P4,Compromised,phase1) :- P1 = 0, P2 = 0, P3 = 0, P4 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,P4,Compromised,phase2) :- P1 = 1, P2 = 0, P3 = 0, P4 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,P4,Compromised,phase3) :- P1 = 1, P2 = 1, P3 = 0, P4 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,P4,Compromised,phase4) :- P1 = 1, P2 = 1, P3 = 1, P4 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,P4,Compromised,phase5) :- Compromised = 1.

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

privileged_port(P) :- P =< 1023.
non_privileged_port(P) :- P >= 1024, P =< 65535.

% Vulnerability knowledge

sadmind_port(111).
telnet(23).

sadmind_req(SrcO,DstO,DPort,Proto) :-
    udp_req(SrcO,DstO,Proto),
    (sadmind_port(DPort); non_privileged_port(DPort)).

telnet_req(SrcO,DstO,DPort,Proto) :-
    tcp_req(SrcO,DstO,Proto),
    telnet(DPort).

privileged_action(DPort,Proto) :-
    tcp(Proto),
    privileged_port(DPort).

mstream(SrcO,DstO,Proto) :-
    ext_orig(SrcO),
    ext_resp(DstO),
    tcp(Proto).

% DARPA MSA phase rules

phase_rule(SrcO,DstO,DPort,Proto,phase1) :-
    icmp_req(SrcO,DstO,Proto).

phase_rule(SrcO,DstO,DPort,Proto,phase2) :-
    sadmind_req(SrcO,DstO,DPort,Proto) ;
    icmp_resp(SrcO,DstO,Proto).

phase_rule(SrcO,DstO,DPort,Proto,phase3) :-
    sadmind_req(SrcO,DstO,DPort,Proto) ;
    telnet_req(SrcO,DstO,DPort,Proto).

phase_rule(SrcO,DstO,DPort,Proto,phase4) :-
    telnet_req(SrcO,DstO,DPort,Proto) ;
    privileged_action(DPort,Proto).

phase_rule(SrcO,DstO,DPort,Proto,phase5) :-
    mstream(SrcO,DstO,Proto).

% Multi-step attack logic

multi_step(X,P1,P2,P3,P4,Compromised,SrcO,DstO,DPort,Proto,Phase) :-
    valid_phase_progression(P1,P2,P3,P4,Compromised,Phase),
    phase_rule(SrcO,DstO,DPort,Proto,Phase),
    msa(X,attack).

multi_step(X,P1,P2,P3,P4,Compromised,SrcO,DstO,DPort,Proto,benign) :-
    \+ (
        phase(Phase),
        multi_step(X,P1,P2,P3,P4,Compromised,SrcO,DstO,DPort,Proto,Phase)
    ).