function steps(n){ let c = 0; while (n !== 1){ n = (n % 2 === 0) ? n/2 : 3*n+1; c++; } return c; }
let total = 0;
for (let i = 1; i <= 1000000; i++) total += steps(i);
console.log(total);
