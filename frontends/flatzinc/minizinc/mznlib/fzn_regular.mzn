predicate qayd_regular(
  array [int] of var int: x,
  int: Q,
  int: S,
  array [int] of int: d,
  int: q0,
  set of int: F,
);

predicate fzn_regular(
  array [int] of var int: x,
  int: Q,
  int: S,
  array [int, int] of int: d,
  int: q0,
  set of int: F,
) = qayd_regular(x, Q, S, [d[i, j] | i in index_set_1of2(d), j in index_set_2of2(d)], q0, F);
