function build(n){ let a = null; while (n){ a = {v:n, t:a}; n--; } return a; }
function suml(xs){ let acc = 0; while (xs){ acc += xs.v; xs = xs.t; } return acc; }
console.log(suml(build(20000000)));
