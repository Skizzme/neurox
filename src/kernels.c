//void atomicAdd_g_f(volatile __global float *addr, float val) {
//    union {
//        uint u32;
//        float f32;
//    } next, expected, current;
//    current.f32 = *addr;
//    do {
//        expected.f32 = current.f32;
//        next.f32 = expected.f32 + val;
//        current.u32 = atom_cmpxchg((volatile __global uint *)addr, expected.u32, next.u32);
//    } while( current.u32 != expected.u32 );
//}
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

float activate(float val) {
//    printf("v: %f, %f\n", max(val, 0.0f), val);
//    return max(val, 0.0f);
//    if (val > 0.0) { return val; } else { return 0.02*val; }
//    return val;
//    return 1 / (1 + exp(-val));
//    return 1 / (1 + exp(-val)) * 2 - 1;
//    return 1 / (1 + exp(-val*4)) * 2 - 1;

    float exp_p = exp(val);
    float exp_n = exp(-val);
    return (exp_p - exp_n) / (exp_p + exp_n);

//    return sin(val);
}

float activate_derivative(float value) {
//    value = activate(value);
//    if (value > 0) { return 1.0; } else { return 0.0; }
//    if (value > 0) { return 1.0; } else { return 0.02; }
//    return 1.0;
//    return exp(value) / pow(exp(value) + 1.0, 2.0);
//    return (2.0*exp(value)) / pow(exp(value) + 1.0, 2.0);
//    return (8.0*exp(4*value)) / pow(exp(4*value) + 1.0, 2.0);

    float activated = activate(value);
    return 1 - (activated * activated);

//    return cos(value);
}


__kernel void set_biases(__global float* buffer, __constant float* biases, int offset) {
    int i = get_global_id(0);
    buffer[i] = biases[i+offset];
}

__kernel void random_buf(__global float* buffer, ulong randoms, float div) {
    int i = get_global_id(0);
    ulong result = (((randoms + i*0xFF9D2D) * 0x5DEECE66DL + 0xBL) & ((1L << 48) -1)) >> 16;
    float res = ((float) result) / 4294967295.0;
    // * 2.0 - 1.0 // for -1.0 to 1.0 values
    res = (res * 2.0 - 1.0) / div; // This value division depends on the network size. If it's a big network it must be smaller, smaller network it must be larger.
//    printf("RES: %f\n", res);
    buffer[i] = res;
}

__kernel void activation(__global float* values, __global float* target) {
    int i = get_global_id(0);
    target[i] = activate(values[i]);
}

__kernel void cost(__constant float* values, __constant float* target, __global float* output) {
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

__kernel void activate_and_error_derivative_calc(__global float* values, __global float* desired, __global float* out) {
    int i = get_global_id(0);
    out[i] = error_derivative(activate(values[i]), desired[i]);
}

__kernel void forward(
    int apply_activations_in,
    ulong input_length,
    ulong layer_len,
    ulong weights_offset,
    ulong biases_offset,
    __constant float* weights,
    __constant float* biases,
    __constant float* input,
    __constant float* activated_input,
    __global float* output,
    __global float* activated_output,
    ulong batched_offset_in,
    ulong batched_offset_out
) {
    int x = get_global_id(0); // out dims
    int y = get_global_id(1); // in dims
    int w_ind = (input_length * x) + y + weights_offset;
    int wg_size_x = get_local_size(0);
    int wg_size_y = get_local_size(1);
    int wg_x = get_local_id(0);
    int wg_y = get_local_id(1);

    __local float local_outputs[128][128];

    int batched_x = x + batched_offset_out;
    int batched_y = y + batched_offset_in;

    float in = input[batched_y];
    if (apply_activations_in == 1) {
        in = activated_input[batched_y];
    }

    local_outputs[wg_y][wg_x] = in*weights[w_ind];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (wg_y == 0) {
        float sum = 0;
        for (int i = 0; i < wg_size_y; i++) {
            sum += local_outputs[i][wg_x];
        }
        atomicAdd_g_f(&output[batched_x], sum);
    }

    if (y == 0) {
        atomicAdd_g_f(&output[batched_x], biases[x+biases_offset]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    activated_output[batched_x] = activate(output[batched_x]);
}

__kernel void backward(
    ulong input_length,
    ulong weights_offset,
    ulong biases_offset,
    float learn_rate,
    __constant float* inputs,
    __constant float* layer_output,
    __constant float* sensitivities,
    __constant float* weights,
    __constant float* biases,
    __global float* weight_mods,
    __global float* bias_mods,
    __global float* gradients_out,
    ulong batched_offset_in,
    ulong batched_offset_out
) {
    int x = get_global_id(0); // out dims
    int y = get_global_id(1); // in dims
    int wg_size_x = get_local_size(0);
    int wg_size_y = get_local_size(1);
    int wg_x = get_local_id(0);
    int wg_y = get_local_id(1);
    ulong weight_index = (input_length * x) + y + weights_offset;
    ulong bias_index = x+biases_offset;

    __local float local_gradients[128][128];

    int batched_x = x + batched_offset_out;
    int batched_y = y + batched_offset_in;

    float gradient = activate_derivative(layer_output[batched_x]) * sensitivities[batched_x];

    weight_mods[weight_index] -= (learn_rate * inputs[batched_y] * gradient);

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
        atomicAdd_g_f(&gradients_out[batched_y], sum);
    }

}

// gradient = input * error_derivative(actual_output - desired_output)
// new_weight = original_weight - learn_rate * gradient