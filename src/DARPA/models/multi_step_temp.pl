% --- Neural network definitions ---
nn(phase_1_net, [tensor(train, I)], Y, [0,1]) :: s1(I, Y).
nn(phase_2_net, [tensor(train, I)], Y, [0,1]) :: s2(I, Y).
nn(phase_3_net, [tensor(train, I)], Y, [0,1]) :: s3(I, Y).
nn(phase_4_net, [tensor(train, I)], Y, [0,1]) :: s4(I, Y).
nn(phase_5_net, [tensor(train, I)], Y, [0,1]) :: s5(I, Y).

% --- Map outputs ---
flow_attack(I,1) :- s1(I,1).
flow_attack(I,2) :- s2(I,1), \+ s1(I,1).
flow_attack(I,3) :- s3(I,1), \+ s1(I,1), \+ s2(I,1).
flow_attack(I,4) :- s4(I,1), \+ s1(I,1), \+ s2(I,1), \+ s3(I,1).
flow_attack(I,5) :- s5(I,1), \+ s1(I,1), \+ s2(I,1), \+ s3(I,1), \+ s4(I,1).
flow_attack(I,0) :-
    \+ s1(I,1), \+ s2(I,1), \+ s3(I,1), \+ s4(I,1), \+ s5(I,1).


% --- Optional: multi_step_attack predicate for general attack detection ---
% This keeps the old "at least 2 phases active" logic
count_true(X,N) :-
    N is
      (s1(X,1) -> 1 ; 0) +
      (s2(X,1) -> 1 ; 0) +
      (s3(X,1) -> 1 ; 0) +
      (s4(X,1) -> 1 ; 0) +
      (s5(X,1) -> 1 ; 0).

multi_step_attack(X) :-
    count_true(X,N),
    N >= 2.
