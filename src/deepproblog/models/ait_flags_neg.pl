nn(net1, [X], Z, [benign, attack]) :: scan(X, Z).
nn(net2, [X], Z, [benign, attack]) :: exploit(X, Z).
nn(net3, [X], Z, [benign, attack]) :: priv_esc(X, Z).
nn(net4, [X], Z, [benign, attack]) :: data_exfil(X, Z).

% Valid phase progressions

attack_phase(phase1).
attack_phase(phase2).
attack_phase(phase3).
attack_phase(phase4).

valid_phase_progression(P1,P2,P3,Compromised,phase1) :- P1 = 0, P2 = 0, P3 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,Compromised,phase2) :- P1 = 1, P2 = 0, P3 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,Compromised,phase3) :- P1 = 1, P2 = 1, P3 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,Compromised,phase4) :- Compromised = 1.

% Phase rules

phase_rule(X,phase1) :-
    scan(X,attack).

phase_rule(X,phase2) :-
    exploit(X,attack).

phase_rule(X,phase3) :-
    priv_esc(X,attack).

phase_rule(X,phase4) :-
    data_exfil(X,attack).

% Multi-step attack logic

multi_step(X,P1,P2,P3,Compromised,_,_,_,_,Phase) :-
    valid_phase_progression(P1,P2,P3,Compromised,Phase),
    phase_rule(X,Phase).

multi_step(X,P1,P2,P3,Compromised,_,_,_,_,benign) :-
    \+ (
        attack_phase(Phase),
        multi_step(X,P1,P2,P3,Compromised,_,_,_,_,Phase)
    ).