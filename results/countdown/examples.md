# Countdown: before and after GRPO

Held-out problems with the greedy completion before training (step 0) and after the last step, verbatim from `eval_before.jsonl` and `eval_after.jsonl`. The first few problems of each kind are shown whatever the outcome. Completions longer than 1200 characters are cut and marked.

## Using the numbers [18, 21, 4], write an arithmetic expression that equals 25. You can use + - * / and parentheses, and each number at most once. The answer must be the expression itself, not its result.

Ground truth: `{"numbers": [18, 21, 4], "target": 25}`

**Before** (bad format, truncated, reward 0.0, 256 tokens)

```text
To solve the problem of finding an arithmetic expression that equals 25 using the numbers 18, 21, and 4, we need to carefully consider the operations and their placements. Let's break it down step by step.

1. **Identify the target value**: We need to find an expression that equals 25.

2. **Consider the operations**: We can use addition, subtraction, multiplication, and division. However, since we need to use each number exactly once and the expression must equal 25, we need to think creatively.

3. **Try different combinations**: Let's start by trying to use the numbers in a way that might lead to a sum close to 25.

4. **Use subtraction and multiplication**: We can use subtraction and multiplication to get close to 25. Let's try:
   - Subtract 1 from 21: \(21 - 1 = 20\)
   - Multiply 21 by 2: \(21 \times 2 = 42\)
   - Add 4 to 42: \(42 + 4 = 46\)
   - Subtract 1 from 46: \(46 - 1 =
```

**After** (wrong answer, reward 0.1, 53 tokens)

```text
<think> use the numbers 18, 21, 4 and the operations +, -, *, / to get 25 </think> <answer> (18 + 21 - 4) * 5 </answer>
```

## Using the numbers [19, 18, 7], write an arithmetic expression that equals 145. You can use + - * / and parentheses, and each number at most once. The answer must be the expression itself, not its result.

Ground truth: `{"numbers": [19, 18, 7], "target": 145}`

**Before** (bad format, truncated, reward 0.0, 256 tokens)

```text
To solve the problem of finding an arithmetic expression that equals 145 using the numbers 19, 18, and 7, we need to carefully consider the operations and their placements. Let's break it down step by step.

1. **Identify the target value**: We need to reach 145 using the numbers 19, 18, and 7.

2. **Consider the operations**: We can use addition, subtraction, multiplication, and division. Let's start by trying to use the largest numbers first and work our way down.

3. **First try**: Let's try to use the largest number, 19, and see if we can get close to 145.
   - \( 19 + 18 = 37 \)
   - Now we have 37 and 7 left. We can use subtraction to get closer.
   - \( 37 - 7 = 30 \)
   - Now we have 30 and 18 left. We can use subtraction again.
   - \( 30 - 18 = 12 \)
   - Now we have 12 and 7 left. We can use subtraction again
```

**After** (wrong answer, reward 0.1, 58 tokens)

```text
<think> use the numbers 19, 18, 7 and the operations -, *, /, and parentheses to get 145 </think> <answer> (19 - 18) * 7 = 145 </answer>
```

## Using the numbers [5, 6, 6], write an arithmetic expression that equals 36. You can use + - * / and parentheses, and each number at most once. The answer must be the expression itself, not its result.

Ground truth: `{"numbers": [5, 6, 6], "target": 36}`

**Before** (bad format, truncated, reward 0.0, 256 tokens)

```text
To solve the problem of finding an arithmetic expression that equals 36 using the numbers 5, 6, and 6 exactly once, we need to carefully consider the operations and their placements. Let's break it down step by step.

1. **Identify the target value**: We need to reach 36 using the numbers 5, 6, and 6 exactly once.

2. **Consider different combinations**:
   - Let's start by trying to use the largest numbers first to maximize the value.
   - One possible combination is \(6 + 6 + 6 = 18\), but this doesn't work because we need to use 5 and 6 exactly once.

3. **Try other combinations**:
   - Let's try \(6 + 6 + 5 = 17\), but this also doesn't work because we need to use 5 and 6 exactly once.
   - Let's try \(6 + 5 + 6 = 17\), but this also doesn't work because we need to use 5 and 6 exactly once.
   - Let's try \(6 + 6 + 5 = 17\), but this also doesn't work because we need
```

**After** (wrong answer, reward 0.1, 45 tokens)

```text
<think> use the numbers 5, 6, 6 to form an expression that equals 36 </think> <answer> (5 + 6) * 6 - 5 </answer>
```

## Using the numbers [11, 11, 4], write an arithmetic expression that equals 88. You can use + - * / and parentheses, and each number at most once. The answer must be the expression itself, not its result.

Ground truth: `{"numbers": [11, 11, 4], "target": 88}`

**Before** (bad format, truncated, reward 0.0, 256 tokens)

```text
To solve the problem of finding an arithmetic expression that equals 88 using the numbers 11, 11, and 4, we need to carefully consider the operations and their placements. Let's break it down step by step.

1. **Identify the target value**: We need to reach 88 using the numbers 11, 11, and 4.

2. **Consider the operations**: We can use addition, subtraction, multiplication, and division. Let's start by trying to use the numbers in a way that might lead us to 88.

3. **First try**: Let's try to use the numbers in a way that might lead us to 88. One possible approach is to use the numbers in a way that might help us reach 88. Let's try to use the numbers in a way that might lead us to 88.

4. **First try**: Let's try to use the numbers in a way that might lead us to 88. One possible approach is to use the numbers in a way that might lead us to 88. Let's try to use the numbers in a way that might lead us to 88.

5. **First try**: Let's
```

**After** (wrong answer, reward 0.1, 50 tokens)

```text
<think> use the numbers 11, 11, 4 to form an expression that equals 88 </think> <answer> (11 + 11) * 4 = 88 </answer>
```

## Using the numbers [6, 24, 4], write an arithmetic expression that equals 36. You can use + - * / and parentheses, and each number at most once. The answer must be the expression itself, not its result.

Ground truth: `{"numbers": [6, 24, 4], "target": 36}`

**Before** (bad format, truncated, reward 0.0, 256 tokens)

```text
To solve the problem of finding an arithmetic expression that equals 36 using the numbers 6, 24, and 4, we need to carefully consider the operations and their placements. Let's break it down step by step.

1. **Identify the target value**: We need to find an expression that equals 36.

2. **Consider the operations**: We can use addition, subtraction, multiplication, and division. However, since we have three numbers and we need to get to 36, we need to use the operations in a way that allows us to reach 36.

3. **Try different combinations**: Let's start by trying to use the numbers in a way that might lead us to 36. One possible approach is to use the numbers in a way that allows us to reach 36 by combining them in a strategic manner.

4. **Use the numbers in a strategic way**: Let's try to use the numbers in a way that allows us to reach 36. One possible approach is to use the numbers in a way that allows us to reach 36 by combining them in a strategic manner.

5. **Calculate the expression**: Let's calculate the expression step by step to see if we can reach
```

**After** (wrong answer, reward 0.1, 51 tokens)

```text
<think> use the numbers 6, 24, 4 and the operations +, -, *, / to get 36 </think> <answer> (6 + 24) * 4 / 3 </answer>
```

# Problems the trained model got right

Every other problem the final greedy policy solved, beyond the ones above.

## Using the numbers [20, 20, 5], write an arithmetic expression that equals 200. You can use + - * / and parentheses, and each number at most once. The answer must be the expression itself, not its result.

Ground truth: `{"numbers": [20, 20, 5], "target": 200}`

**Before** (bad format, truncated, reward 0.0, 256 tokens)

```text
To solve the problem of finding an arithmetic expression that equals 200 using the numbers 20, 20, and 5, we need to carefully consider the operations and their placements. Here's a step-by-step approach:

1. **Identify the target value**: We need to reach 200 from the numbers 20, 20, and 5.

2. **Consider the operations**: We can use addition, subtraction, multiplication, and division. Let's start by trying different combinations.

3. **Try different combinations**:
   - Let's start with the numbers 20 and 5.
   - We need to use the numbers 20 and 20 to reach 200. One way to do this is to use subtraction and multiplication.

4. **Construct the expression**:
   - We can use the expression \(20 \times 20 - 20\).
   - Let's break it down:
     - First, calculate \(20 \times 20 = 400\).
     - Then, subtract 20 from 400 to get 380.
     - Finally, subtract 20 from 380 to
```

**After** (correct, reward 1.1, 51 tokens)

```text
<think> use the numbers 20, 20, 5 and the operations +, -, *, / to get 200 </think> <answer> (20 + 20) * 5 </answer>
```
