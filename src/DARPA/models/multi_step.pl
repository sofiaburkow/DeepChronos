nn(phase_1_net,[X],Y,[0,1]) :: s1(X,Y).
nn(phase_2_net,[X],Y,[0,1]) :: s2(X,Y).
nn(phase_3_net,[X],Y,[0,1]) :: s3(X,Y).
nn(phase_4_net,[X],Y,[0,1]) :: s4(X,Y).
nn(phase_5_net,[X],Y,[0,1]) :: s5(X,Y).

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
