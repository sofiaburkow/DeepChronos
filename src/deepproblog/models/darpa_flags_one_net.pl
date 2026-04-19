nn(net1, [X], Z, [benign,attack]) :: msa(X,Z).

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

% Multi-step attack logic

multi_step(X,P1,P2,P3,P4,Compromised,_,_,_,_,Phase) :-
    valid_phase_progression(P1,P2,P3,P4,Compromised,Phase),
    msa(X,attack).

multi_step(X,P1,P2,P3,P4,Compromised,_,_,_,_,benign) :-
    msa(X,benign).