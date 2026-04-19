nn(net1, [X], Z, [benign,attack]) :: ping(X,Z).
nn(net2, [X], Z, [benign,attack]) :: probing(X,Z). 
nn(net3, [X], Z, [benign,attack]) :: exploit(X,Z).
nn(net4, [X], Z, [benign,attack]) :: install(X,Z).
nn(net5, [X], Z, [benign,attack]) :: ddos(X,Z).

% Valid phase progressions

attack_phase(phase1).
attack_phase(phase2).
attack_phase(phase3).
attack_phase(phase4).
attack_phase(phase5).

valid_phase_progression(P1,P2,P3,P4,Compromised,phase1) :- P1 = 0, P2 = 0, P3 = 0, P4 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,P4,Compromised,phase2) :- P1 = 1, P2 = 0, P3 = 0, P4 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,P4,Compromised,phase3) :- P1 = 1, P2 = 1, P3 = 0, P4 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,P4,Compromised,phase4) :- P1 = 1, P2 = 1, P3 = 1, P4 = 0, Compromised = 0.
valid_phase_progression(P1,P2,P3,P4,Compromised,phase5) :- Compromised = 1.

% DARPA MSA phase rules

phase_rule(X,phase1) :-
    ping(X,attack).

phase_rule(X,phase2) :-
    probing(X,attack).

phase_rule(X,phase3) :-
    exploit(X,attack).

phase_rule(X,phase4) :-
    install(X,attack).

phase_rule(X,phase5) :-
    ddos(X,attack).

% Multi-step attack logic

multi_step(X,P1,P2,P3,P4,Compromised,_,_,_,_,Phase) :-
    valid_phase_progression(P1,P2,P3,P4,Compromised,Phase),
    phase_rule(X,Phase).

multi_step(X,P1,P2,P3,P4,Compromised,_,_,_,_,benign) :-
    \+ (
        attack_phase(Phase),
        multi_step(X,P1,P2,P3,P4,Compromised,_,_,_,_,Phase)
    ).