nn(net1, [X], Z, [benign, malicious]) :: event(X, Z).

% Attack phase definitions

attack_phase(phase1).
attack_phase(phase2).
attack_phase(phase3).
attack_phase(phase4).
attack_phase(phase5).

next_attack_phase(P1,P2,P3,P4,phase1) :- P1 = 0, P2 = 0, P3 = 0, P4 = 0.
next_attack_phase(P1,P2,P3,P4,phase2) :- P1 = 1, P2 = 0, P3 = 0, P4 = 0.
next_attack_phase(P1,P2,P3,P4,phase3) :- P1 = 1, P2 = 1, P3 = 0, P4 = 0.
next_attack_phase(P1,P2,P3,P4,phase4) :- P1 = 1, P2 = 1, P3 = 1, P4 = 0.
next_attack_phase(P1,P2,P3,P4,phase5) :- P1 = 1, P2 = 1, P3 = 1, P4 = 1.

% Network traffic pattern definitions

home_orig(1).
home_resp(1).
ext_orig(0).
ext_resp(0).

icmp(1).
tcp(6).
udp(17).

sadmind_port(111).
telnet_port(23).
privileged_port(P) :- P =< 1023.
non_privileged_port(P) :- P >= 1024, P =< 65535.

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

sadmind_req(SrcO,DstO,DPort,Proto) :-
    udp_req(SrcO,DstO,Proto),
    (sadmind_port(DPort); non_privileged_port(DPort)).

telnet_req(SrcO,DstO,DPort,Proto) :-
    tcp_req(SrcO,DstO,Proto),
    telnet_port(DPort).

privileged_action(DPort,Proto) :-
    tcp(Proto),
    privileged_port(DPort).

mstream(SrcO,DstO,Proto) :-
    ext_orig(SrcO),
    ext_resp(DstO),
    tcp(Proto).

% Attack phase inference

0.9::phase_soft(SrcO,DstO,DPort,Proto,phase1) :- icmp_req(SrcO,DstO,Proto).
0.1::phase_soft(SrcO,DstO,DPort,Proto,phase1).

0.9::phase_soft(SrcO,DstO,DPort,Proto,phase2) :- sadmind_req(SrcO,DstO,DPort,Proto) ; icmp_resp(SrcO,DstO,Proto).
0.1::phase_soft(SrcO,DstO,DPort,Proto,phase2).

0.9::phase_soft(SrcO,DstO,DPort,Proto,phase3) :- sadmind_req(SrcO,DstO,DPort,Proto) ; telnet_req(SrcO,DstO,DPort,Proto).
0.1::phase_soft(SrcO,DstO,DPort,Proto,phase3).

0.9::phase_soft(SrcO,DstO,DPort,Proto,phase4) :- telnet_req(SrcO,DstO,DPort,Proto) ; privileged_action(DPort,Proto).
0.1::phase_soft(SrcO,DstO,DPort,Proto,phase4).

% 0.9::phase_soft(SrcO,DstO,DPort,Proto,phase5) :- mstream(SrcO,DstO,Proto).
1.0::phase_soft(SrcO,DstO,DPort,Proto,phase5).

multi_step(X,P1,P2,P3,P4,SrcO,DstO,DPort,Proto,NextPhase) :-
    next_attack_phase(P1,P2,P3,P4,NextPhase),
    phase_soft(SrcO,DstO,DPort,Proto,NextPhase),
    event(X, malicious).

multi_step(X,P1,P2,P3,P4,SrcO,DstO,DPort,Proto,benign) :-
    \+ (
        attack_phase(NextPhase),
        multi_step(X,P1,P2,P3,P4,SrcO,DstO,DPort,Proto,NextPhase)
    ).