nn(phase_1_net, [X], Y, [0,1]) :: phase1(X,Y).
nn(phase_2_net, [X], Y, [0,1]) :: phase2(X,Y).
nn(phase_3_net, [X], Y, [0,1]) :: phase3(X,Y).
nn(phase_4_net, [X], Y, [0,1]) :: phase4(X,Y).
nn(phase_5_net, [X], Y, [0,1]) :: phase5(X,Y).

% recon(X,Y) :- phase1(X,Z1), phase2(X,Z2), Z1+Z2 >= Y.

recon(X,Y) :- phase1(X,Y).
recon(X,Y) :- phase2(X,Y).



% 0.7::multi_step(X,Y) :- phase1(X,Z), Z is Y.
% multi_step(X) :- phase1(X,1), phase2(X,1).
% multi_step(X) :- phase2(X,1), phase3(X,1).
% multi_step(X) :- phase3(X,1), phase4(X,1).
% multi_step(X) :- phase4(X,1), phase5(X,1).