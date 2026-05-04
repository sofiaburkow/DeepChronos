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

int_to_int(SrcO, DstO) :- home_orig(SrcO), home_resp(DstO).

% Attack phase inference

0.8::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase1) :- int_to_int(SrcO,DstO), dns_port(DPort).
0.2::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase1).

0.8::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase2) :- int_to_int(SrcO,DstO), tcp(Proto).
0.2::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase2).

0.8::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase3) :- int_to_int(SrcO,DstO), https_port(DPort).
0.2::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase3).

0.8::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase4) :- int_to_int(SrcO,DstO), tcp(Proto), (http_port(DPort) ; https_port(DPort)).
0.2::phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,phase4).

multi_step(_,P1,P2,P3,SrcO,DstO,DPort,Proto,ExfilSig,NextPhase) :-
    next_attack_phase(P1,P2,P3,NextPhase),
    phase_soft(SrcO,DstO,DPort,Proto,ExfilSig,NextPhase).

multi_step(_,P1,P2,P3,SrcO,DstO,DPort,Proto,ExfilSig,benign) :-
    \+ (
        attack_phase(NextPhase),
        multi_step(_,P1,P2,P3,SrcO,DstO,DPort,Proto,ExfilSig,NextPhase)
    ).