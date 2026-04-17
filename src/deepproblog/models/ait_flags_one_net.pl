nn(net1, [X], Z, [benign, attack]) :: msa(X, Z).

t(1.0)::valid_phase(phase1,_,0,0,0).
t(1.0)::valid_phase(phase2,1,_,0,0).
t(1.0)::valid_phase(phase3,1,1,_,0).
t(1.0)::valid_phase(phase4,1,1,1,_).

multi_step(P1,P2,P3,P4,X,Phase) :-
    msa(X,attack),
    valid_phase(Phase,P1,P2,P3,P4).

multi_step(P1,P2,P3,P4,X,benign) :-
    \+ multi_step(P1,P2,P3,P4,X,phase1),
    \+ multi_step(P1,P2,P3,P4,X,phase2),
    \+ multi_step(P1,P2,P3,P4,X,phase3),
    \+ multi_step(P1,P2,P3,P4,X,phase4).