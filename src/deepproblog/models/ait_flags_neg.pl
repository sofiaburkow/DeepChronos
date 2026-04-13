nn(net1, [X], Z, [benign, attack]) :: msa(phase1, X, Z).
nn(net2, [X], Z, [benign, attack]) :: msa(phase2, X, Z).
nn(net3, [X], Z, [benign, attack]) :: msa(phase3, X, Z).
nn(net4, [X], Z, [benign, attack]) :: msa(phase4, X, Z).

t(1.0)::valid_phase(phase1,_,0,0,0).
t(1.0)::valid_phase(phase2,1,_,0,0).
t(1.0)::valid_phase(phase3,1,1,_,0).
t(1.0)::valid_phase(phase4,1,1,1,_).

multi_step(P1,P2,P3,P4,X,Phase) :-
    msa(Phase, X, attack),
    valid_phase(Phase,P1,P2,P3,P4).

multi_step(P1,P2,P3,P4,X,benign) :-
    \+ multi_step(P1,P2,P3,P4,X,phase1),
    \+ multi_step(P1,P2,P3,P4,X,phase2),
    \+ multi_step(P1,P2,P3,P4,X,phase3),
    \+ multi_step(P1,P2,P3,P4,X,phase4).