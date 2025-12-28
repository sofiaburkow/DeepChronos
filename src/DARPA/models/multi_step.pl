nn(phase_1_net,[X],Y,[0,1]) :: s1(X,Y1).
nn(phase_2_net,[Y1],Y,[0,1]) :: s2(X,Y2) :- s1(X,Y1).
nn(phase_3_net,[Y2],Y,[0,1]) :: s3(X,Y3) :- s2(X,Y2).
nn(phase_4_net,[Y3],Y,[0,1]) :: s4(X,Y4) :- s3(X,Y3).
nn(phase_5_net,[Y4],Y,[0,1]) :: s5(X,Y5) :- s4(X,Y4).

multi_step_attack(X) :- 
    s1(X,1), s2(X,1), s3(X,1), s4(X,1), s5(X,1),
    Sum is v1 + v2 + v3 + v4 + v5,
    Sum >= 2. % require at least 2 positives