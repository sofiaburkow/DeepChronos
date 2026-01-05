nn(phase_1_net, [X], Y, [0,1]) :: phase1(X,Y).
nn(phase_2_net, [X], Y, [0,1]) :: phase2(X,Y).
nn(phase_3_net, [X], Y, [0,1]) :: phase3(X,Y).
nn(phase_4_net, [X], Y, [0,1]) :: phase4(X,Y).
nn(phase_5_net, [X], Y, [0,1]) :: phase5(X,Y).

multi_step(X) :- phase1(X,1).
% multi_step(X) :- phase1(X,1), phase2(X,1).
% multi_step(X) :- phase2(X,1), phase3(X,1).
% multi_step(X) :- phase3(X,1), phase4(X,1).
% multi_step(X) :- phase4(X,1), phase5(X,1).