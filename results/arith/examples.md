# Arithmetic: before and after GRPO

Held-out problems with the greedy completion before training (step 0) and after the last step, verbatim from `eval_before.jsonl` and `eval_after.jsonl`. The first few problems of each kind are shown whatever the outcome. Completions longer than 1200 characters are cut and marked.

## What is 361 + 607? Give the answer as a single integer.

Ground truth: `{"answer": 968}`

**Before** (bad format, reward 0.0, 148 tokens)

```text
To solve the problem of adding 361 and 607, we can use the standard addition method. Here's the step-by-step process:

1. Start with the numbers: 361 and 607.
2. Add the digits in the ones place: 1 + 7 = 8.
3. Add the digits in the tens place: 6 + 0 = 6.
4. Write down the sum of the digits in the ones place: 8.
5. Write down the sum of the digits in the tens place: 6.
6. Combine the results from the ones and tens places: 86.

So, the final answer is 86.
```

**After** (correct, reward 1.1, 31 tokens)

```text
<think> 361 + 607 = 968 </think> <answer> 968 </answer>
```

## What is 7219 - 196? Give the answer as a single integer.

Ground truth: `{"answer": 7023}`

**Before** (bad format, reward 0.0, 116 tokens)

```text
To solve the problem \(7219 - 196\), we can break it down into simpler steps:

1. **Subtract the smaller number from the larger number:**
   \[
   7219 - 196 = 6923
   \]

2. **Check if there's any remainder:**
   Since 7219 is greater than 196, there is no remainder to consider.

Therefore, the final answer is:
\[
\boxed{6923}
\]
```

**After** (wrong answer, reward 0.1, 34 tokens)

```text
<think> 7219 - 196 = 6023 </think> <answer> 6023 </answer>
```

## What is 955 - 8090? Give the answer as a single integer.

Ground truth: `{"answer": -7135}`

**Before** (bad format, reward 0.0, 190 tokens)

```text
To solve the subtraction problem 955 - 8090, we can follow these steps:

1. Start with the number 955.
2. Subtract 8090 from 955.

Let's do the subtraction step by step:

1. Subtract the units place: 5 - 0 = 5
2. Subtract the tens place: 5 - 9 (since we can't subtract 9 from 5, we need to borrow 1 from the hundreds place)
   - 5 becomes 15
   - 15 - 9 = 6
3. Subtract the hundreds place: 9 - 8 = 1
4. Subtract the thousands place: 0 - 0 = 0

So, 955 - 8090 = 425

Therefore, the final answer is 425.
```

**After** (correct, reward 1.1, 34 tokens)

```text
<think> 955 - 8090 = -7135 </think> <answer> -7135 </answer>
```

## What is 85 * 50? Give the answer as a single integer.

Ground truth: `{"answer": 4250}`

**Before** (bad format, truncated, reward 0.0, 256 tokens)

```text
To solve the problem of multiplying 85 by 50, we can use the standard multiplication method. Here's the step-by-step process:

1. **Place Value Breakdown:**
   - 85 can be broken down into 80 and 5.
   - 50 can be broken down into 50.

2. **Multiply 80 by 50:**
   - First, multiply 80 by 5: \(80 \times 5 = 400\).
   - Then, add the result to the product of 80 and 50: \(400 + 400 = 800\).

3. **Multiply 50 by 50:**
   - First, multiply 50 by 5: \(50 \times 5 = 250\).
   - Then, add the result to the product of 50 and 50: \(250 + 250 = 500\).

4. **Combine the Results:**
   - Add the two results from the previous steps: \(800 + 500 = 1300\).

Therefore,
```

**After** (correct, reward 1.1, 31 tokens)

```text
<think> 85 * 50 = 4250 </think> <answer> 4250 </answer>
```

## What is 126 + 490? Give the answer as a single integer.

Ground truth: `{"answer": 616}`

**Before** (bad format, reward 0.0, 73 tokens)

```text
To solve the problem of adding 126 and 490, we can follow these steps:

1. Start with the numbers: 126 and 490.
2. Add the numbers together: 126 + 490 = 616.

So, the final answer is 616.
```

**After** (correct, reward 1.1, 31 tokens)

```text
<think> 126 + 490 = 616 </think> <answer> 616 </answer>
```

## What is 64 * 77? Give the answer as a single integer.

Ground truth: `{"answer": 4928}`

**Before** (bad format, reward 0.0, 176 tokens)

```text
To solve the problem of multiplying 64 by 77, we can use the standard multiplication method. Here's the step-by-step process:

1. **Set up the multiplication**: Write 64 in the top row and 77 in the second row.
2. **Multiply each digit**: Start from the rightmost digit (the units place) and move to the left.
3. **Add the products**: Add the products of the digits together.

Let's do this step by step:

```
  64
× 77
------
  448
  448
+ 448
------
  896
```

So, \( 64 \times 77 = 4992 \).

Therefore, the final answer is \( 4992 \).
```

**After** (wrong answer, reward 0.1, 31 tokens)

```text
<think> 64 * 77 = 4848 </think> <answer> 4848 </answer>
```
