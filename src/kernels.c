void atomicAdd_g_f(volatile __global float *addr, float val)
{
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}
void atomicAdd_g_f_l(volatile __local float *addr, float val)
{
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg( (volatile __local unsigned int *)addr, expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

float activate(float val, ulong activation) {
    if (activation == 1) {
        // ReLU
        if (val > 0.0) { return val; } else { return 0; }
    }
    else if (activation == 2) {
        // TanH
        float exp_p = exp(val);
        float exp_n = exp(-val);
        return (exp_p - exp_n) / (exp_p + exp_n);
    } else {
        // Linear
        return val;
    }
}

float activate_derivative(float value, ulong activation) {
    if (activation == 1) {
        // ReLU
        if (value > 0) { return 1.0; } else { return 0.0; }
    }
    else if (activation == 2) {
        // TanH
        float activated = activate(value, activation);
        return 1.0 - (activated * activated);
    } else {
        // Linear
        return 1.0;
    }
}


__kernel void set_biases(__global float* buffer, __constant float* biases, int offset) {
    int i = get_global_id(0);
    buffer[i] = biases[i+offset];
}

__kernel void random_buf(__global float* buffer, ulong randoms, float div) {
    int i = get_global_id(0);
    ulong result = (((randoms + i*0xFF9D2D) * 0x5DEECE66DL + 0xBL) & ((1L << 48) -1)) >> 16;
    float res = ((float) result) / 4294967295.0;
    res = (res * 2.0 - 1.0) / div; // This value division depends on the network size. If it's a big network it must be smaller, smaller network it must be larger.
    buffer[i] = res;
}

__kernel void activation(__global float* values, __global float* target, ulong activation) {
    int i = get_global_id(0);
    target[i] = activate(values[i], activation);
}

__kernel void cost(__constant float* values, __constant float* target, __global float* output, ulong error_func) {
    int i = get_global_id(0);
    atomicAdd_g_f(&output[0], pow(values[i]-target[i], 2.0f));
//    error[i] = target[i]-values[i];
}

float error_derivative(float actual, float desired) {
    return actual - desired; // category
//    return 2.0 * (actual - desired); // prediction
}

__kernel void multiply(__global float* first, __global float* second, __global float* target) {
    int index = get_global_id(0);
    target[index] = first[index] * second[index];
}

__kernel void mult_second_and_add(__global float* first, __global float* second, float mult) {
    int index = get_global_id(0);
    first[index] = first[index] + (second[index] * mult);
}

__kernel void multiply_single(__global float* first, float second, __global float* target) {
     int index = get_global_id(0);
     target[index] = first[index] * second;
 }

__kernel void flat_combine_matrix(__global float* matrix, __global float* out, int x_len) {
    int x = get_global_id(0);
    atomicAdd_g_f(&out[x], matrix[(x_len*get_global_id(1))+x]);
}

__kernel void list_divide_inplace(__global float* top, float bottom) {
    int i = get_global_id(0);
    top[i] = top[i]/bottom;
}

__kernel void activate_and_error_derivative_calc(__global float* values, __global float* desired, __global float* out, ulong activation) {
    int i = get_global_id(0);
    out[i] = error_derivative(activate(values[i], activation), desired[i]);
}

__kernel void forward(
    ulong activation,
    ulong input_length,
    ulong layer_len,
    __constant float* weights,
    __constant float* biases,
    __constant float* input,
    __global float* output,
    __global float* activated_output,
    ulong offset_in,
    ulong offset_out
) {
    int x = get_global_id(0); // out dims
    int y = get_global_id(1); // in dims
    int w_ind = (input_length * x) + y;
    int wg_size_x = get_local_size(0);
    int wg_size_y = get_local_size(1);
    int wg_x = get_local_id(0);
    int wg_y = get_local_id(1);

    __local float local_outputs[64][64];

    int offset_x = x + offset_out;
    int offset_y = y + offset_in;

    float in = input[offset_y];

    // calculate each local multiplication in a local array
    local_outputs[wg_y][wg_x] = in*weights[w_ind];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (wg_y == 0) {
        // sum each local array together into the global array to spend less time waiting for the atomic to unlock
        float sum = 0;
        for (int i = 0; i < wg_size_y; i++) {
            sum += local_outputs[i][wg_x];
        }
        atomicAdd_g_f(&output[offset_x], sum);
    }

    if (y == 0) {
        atomicAdd_g_f(&output[offset_x], biases[x]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    activated_output[offset_x] = activate(output[offset_x], activation);
}

__kernel void backward(
    ulong activation,
    ulong input_length,
    float learn_rate,
    __constant float* inputs,
    __constant float* layer_output,
    __constant float* sensitivities,
    __constant float* weights,
    __constant float* biases,
    __global float* weight_mods,
    __global float* bias_mods,
    __global float* gradients_out,
    ulong offset_in,
    ulong offset_sens,
    ulong offset_out
) {
    int x = get_global_id(0); // out dims
    int y = get_global_id(1); // in dims
    int wg_size_x = get_local_size(0);
    int wg_size_y = get_local_size(1);
    int wg_x = get_local_id(0);
    int wg_y = get_local_id(1);
    ulong weight_index = (input_length * x) + y;
    ulong bias_index = x;

    __local float local_gradients[128][128];

    int offset_x = x + offset_out;
    int offset_y = y + offset_in;

    float gradient = activate_derivative(layer_output[offset_x], activation) * sensitivities[offset_x];

    weight_mods[weight_index] -= (learn_rate * inputs[offset_y] * gradient);

    if (y == 0) {
        bias_mods[bias_index] -= (learn_rate * gradient);
    }

    local_gradients[wg_y][wg_x] = weights[weight_index] * gradient;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (wg_x == 0) {
        float sum = 0;
        for (int i = 0; i < wg_size_x; i++) {
            sum += local_gradients[wg_y][i];
        }
        atomicAdd_g_f(&gradients_out[y + offset_sens], sum);
    }

}