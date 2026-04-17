nn(net1, [X], Z, [benign, attack]) :: msa(phase1, X, Z).
nn(net2, [X], Z, [benign, attack]) :: msa(phase2, X, Z).
nn(net3, [X], Z, [benign, attack]) :: msa(phase3, X, Z).
nn(net4, [X], Z, [benign, attack]) :: msa(phase4, X, Z).

% Grounding facts

home_orig(1).
home_resp(1).
ext_orig(0).
ext_resp(0).

dns(53).
http(80).
https(443).

tcp(6).

internal_traffic(Src, Dst) :-
    home_orig(Src),
    home_resp(Dst).

% Valid phase flags

t(0.5)::valid_phase(phase1,0,0,0,0).
t(1.0)::valid_phase(phase1,1,0,0,0).
t(1.0)::valid_phase(phase2,1,1,0,0).
t(1.0)::valid_phase(phase3,1,1,1,0).
t(1.0)::valid_phase(phase4,1,1,1,1).

% Phase definitions

t(1.0)::phase_rule(Src,Dst,Port,Proto,phase1) :-
    internal_traffic(Src,Dst),
    dns(Port).

t(1.0)::phase_rule(Src,Dst,Port,Proto,phase2) :-
    internal_traffic(Src,Dst),
    tcp(Proto).

t(1.0)::phase_rule(Src,Dst,Port,Proto,phase3) :-
    internal_traffic(Src,Dst),
    https(Port).

t(1.0)::phase_rule(Src,Dst,Port,Proto,phase4) :-
    tcp(Proto),
    (http(Port);https(Port)).

% Multi-step attack definition

multi_step(P1,P2,P3,P4,X,Src,Dst,Port,Proto,Phase) :-
    msa(Phase,X,attack),
    phase_rule(Src,Dst,Port,Proto,Phase),
    valid_phase(Phase,P1,P2,P3,P4).

multi_step(P1,P2,P3,P4,X,Src,Dst,Port,Proto,benign) :-
    \+ multi_step(P1,P2,P3,P4,X,Src,Dst,Port,Proto,phase1),
    \+ multi_step(P1,P2,P3,P4,X,Src,Dst,Port,Proto,phase2),
    \+ multi_step(P1,P2,P3,P4,X,Src,Dst,Port,Proto,phase3),
    \+ multi_step(P1,P2,P3,P4,X,Src,Dst,Port,Proto,phase4).