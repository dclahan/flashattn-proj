# GPU project - implementation of FlashAttention Algorithm

## Project Description

**implement scaled dot product attention** 
```
(softmax(Q @ K^T * softmax_scale) @ V)
```

*   **Core CUDA Work:** Implement the FlashAttention-v2 algorithm from the ground up in CUDA.
    *   This is non-trivial and will involve careful management of:
        *   Tiling across GPU blocks and warps.
        *   Using shared memory for the "SRAM" portion of the algorithm.
        *   Online softmax computation (a key trick in FlashAttention).
        *   Avoiding synchronization issues.
*   **State-of-the-Art Comparison:** Comparing implementation against:
    1.  A **baseline naive attention** implementation I write (sequential/multicore equivalent).
    2.  The **official FlashAttention code** torch uses flashattn in `torch.nn.functional.scaled_dot_product_attention`. Measure of success is matching or approaching its performance.
*   **Experimental Setup & Analysis (Using Nsight Compute):**
    *   Testing on different GPUs available at NYU (e.g., an older V100 vs. a newer A100) to analyze **scalability**.
    *   Varying problem size (sequence length, batch size, head dimension) to show how performance characteristics change.
    *   Use Nsight Compute attempt to **prove** implementation's efficiency:
        *   Demonstrate reduced global memory transactions (HBM reads/writes).
        *   Show high shared memory utilization and bandwidth.
        *   Analyze warp execution efficiency and occupancy.
<!-- *   **"Lessons Learned":** Conclusion will extract general lessons for optimizing for GPU memory hierarchy, applicable beyond just attention mechanisms. -->

## Timeline + Milestones
1.  **September - October:** Literature survey for GPU algo side and math side. Read the FlashAttention paper (v1 and v2) deeply. Study the relevant mathematical background (external memory models, numerical analysis for softmax).
2.  **October - November:** Implement the naive attention and FlashAttention kernels in CUDA. **This is the heaviest lift.** Start drafting the mathematical proofs.
3.  **November:** Run experiments, profile with Nsight, and collect data. Finalize proofs. Write the final report for the GPU class, ensuring it has all the required sections (Abstract, Intro, Survey, Proposed Idea (your implementation details), Expt. Setup, Analysis, Conclusion).
4.  **End of November:** Submit the GPU project (code + report).
5.  **December:** Create your presentation poster for the Math class focused on the two proofs and the experimental validation from your GPU work. Prepare your 8-slide presentation for the GPU class, focusing on the performance results and lessons learned.

