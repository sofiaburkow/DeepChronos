% Attack phase definitions

attack_phase(phase1).
attack_phase(phase2).
attack_phase(phase3).
attack_phase(phase4).

next_attack_phase(P1,P2,P3,phase1) :- P1 = 0, P2 = 0, P3 = 0.
next_attack_phase(P1,P2,P3,phase2) :- P1 = 1, P2 = 0, P3 = 0.
next_attack_phase(P1,P2,P3,phase3) :- P1 = 1, P2 = 1, P3 = 0.
next_attack_phase(P1,P2,P3,phase4) :- P1 = 1, P2 = 1, P3 = 1.

% Network traffic pattern definitions

home_orig(1).
home_resp(1).
ext_orig(0).
ext_resp(0).

tcp(6).
udp(17).

dns_port(53).
http_port(80).
https_port(443).

% Attack phase inference

0.9::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase1) :- dns_port(DPort), E = 1.
0.1::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase1).

0.9::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase2) :- tcp(Proto).
0.1::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase3).

0.9::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase3) :- https_port(DPort).
0.1::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase3).

0.9::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase4) :- tcp(Proto), (http_port(Port) ; https_port(Port)).
0.1::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase4).

multi_step(_,P1,P2,P3,SrcO,DstO,DPort,Proto,ExfilSig,NextPhase) :-
    next_attack_phase(P1,P2,P3,NextPhase),
    phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,NextPhase).

multi_step(_,P1,P2,P3,SrcO,DstO,DPort,Proto,ExfilSig,benign) :-
    \+ (
        attack_phase(NextPhase),
        multi_step(_,P1,P2,P3,SrcO,DstO,DPort,Proto,ExfilSig,NextPhase)
    ).