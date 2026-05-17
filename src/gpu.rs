use std::fs::{File, OpenOptions};
use std::io;
use std::io::{BufWriter, Write};
use crate::{ITERS, TOTAL_RESOLUTION, X_RESOLUTION, Y_RESOLUTION};
use indicatif::ProgressIterator;
use std::iter::repeat_n;
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::Queue;
use vulkano::shader::ShaderModule;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    }, descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    }, device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline,
        PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
    DeviceSize,
    Validated,
    VulkanError,
    VulkanLibrary,
};

mod mandelbrot {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            };

            layout(set = 0, binding = 1) buffer Mask {
                uint mask[];
            };

            layout(push_constant) uniform PushConstantData {
                uint x_resolution;
                uint y_resolution;
                uint x_block_size;
                uint y_block_size;
                uint x_block_count;
                uint y_block_count;
                uint iters;
                float exp_min;
                float exp_max;
                uint exp_steps;
            } pc;

            bool mandelbrot(float c_re, float c_im, float exp, uint iters) {
                float z_re = c_re;
                float z_im = c_im;
                for (uint i = 0; i < iters; i++) {
                    float z_norm = length(vec2(z_re, z_im));
                    float z_arg = atan(z_im, z_re);
                    float pow_norm = pow(z_norm, exp);
                    float pow_arg = z_arg * exp;
                    float pow_re = pow_norm * cos(pow_arg);
                    float pow_im = pow_norm * sin(pow_arg);
                    z_re = pow_re + c_re;
                    z_im = pow_im + c_im;
                    if (z_re * z_re + z_im * z_im > 4.0) {
                        return false;
                    }
                }
                return true;
            }

            void main() {
                uint x = gl_GlobalInvocationID.x;
                uint y = gl_GlobalInvocationID.y;

                if (x >= pc.x_resolution || y >= pc.y_resolution) return;

                uint block_x = x / pc.x_block_size;
                uint block_y = y / pc.y_block_size;
                uint block_index = block_y * pc.x_block_count + block_x;

                if (mask[block_index] == 0) return;

                uint index = y * pc.x_resolution + x;

                float re = (float(x) / float(pc.x_resolution)) * 4.0 - 2.0;
                float im = (float(y) / float(pc.y_resolution)) * 4.0 - 2.0;

                bool all = true;
                bool any = false;
                for (uint i = 0; i < pc.exp_steps; i++) {
                    float exp = pc.exp_min + (float(i) / float(pc.exp_steps)) * (pc.exp_max - pc.exp_min);
                    bool in_set = mandelbrot(re, im, exp, pc.iters);
                    if (in_set) any = true;
                    else all = false;
                }
                data[index] = (uint(all) << 1) | uint(any);
            }
        ",
    }
}

mod aggregate {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Pixels {
                uint pixels[];
            };

            layout(set = 0, binding = 1) buffer Blocks {
                uint blocks[];
            };

            layout(set = 0, binding = 2) buffer Mask {
                uint mask[];
            };

            layout(push_constant) uniform PushConstantData {
                uint x_resolution;
                uint y_resolution;
                uint x_block_count;
                uint y_block_count;
                uint x_block_size;
                uint y_block_size;
            } pc;

            void main() {
                uint x = gl_GlobalInvocationID.x;
                uint y = gl_GlobalInvocationID.y;

                if (x >= pc.x_block_count || y >= pc.y_block_count) return;

                uint block_x = x;
                uint block_y = y;
                uint block_index = block_y * pc.x_block_count + block_x;

                if (mask[block_index] == 0) {
//                    blocks[block_index] = 2;
                    return;
                }

                uint min_pixel_x = max(int(block_x * pc.x_block_size), 0);
                uint min_pixel_y = max(int(block_y * pc.y_block_size), 0);

                uint max_pixel_x = min(int(pc.x_resolution), int(min_pixel_x + pc.x_block_size));
                uint max_pixel_y = min(int(pc.y_resolution), int(min_pixel_y + pc.y_block_size));

                bool block_any = false;
                bool block_all = true;

                for (uint pixel_x = min_pixel_x; pixel_x < max_pixel_x; pixel_x++) {
                    for (uint pixel_y = min_pixel_y; pixel_y < max_pixel_y; pixel_y++) {
                        uint pixel_index = pixel_y * pc.x_resolution + pixel_x;
                        uint pixel = pixels[pixel_index];
                        if ((pixel & 1) == 1) {
                            block_any = true;
                        }
                        if ((pixel & 2) == 0) {
                            block_all = false;
                        }
                    }
                }

                blocks[block_index] = (uint(block_all) << 1) | uint(block_any);
            }
        ",
    }
}

pub(crate) const X_BLOCK_SIZE: u32 = 8;
pub(crate) const Y_BLOCK_SIZE: u32 = 8;
pub(crate) const TOTAL_BLOCK_SIZE: u32 = X_BLOCK_SIZE * Y_BLOCK_SIZE;

pub(crate) const X_BLOCK_COUNT: u32 = X_RESOLUTION.div_ceil(X_BLOCK_SIZE);
pub(crate) const Y_BLOCK_COUNT: u32 = Y_RESOLUTION.div_ceil(Y_BLOCK_SIZE);
pub(crate) const TOTAL_BLOCK_COUNT: u32 = X_BLOCK_COUNT * Y_BLOCK_COUNT;

fn allocate_u32_buffer(
    allocator: Arc<StandardMemoryAllocator>,
    len: DeviceSize,
) -> Subbuffer<[u32]> {
    Buffer::new_unsized::<[u32]>(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        len,
    )
    .unwrap()
}

fn allocate_u32_buffer_with(
    allocator: Arc<StandardMemoryAllocator>,
    data: impl ExactSizeIterator<Item = u32>,
) -> Subbuffer<[u32]> {
    Buffer::from_iter(
        allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        data,
    )
    .unwrap()
}

fn load_library() -> Arc<VulkanLibrary> {
    VulkanLibrary::new().unwrap()
}

fn create_instance(library: Arc<VulkanLibrary>) -> Arc<Instance> {
    Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap()
}

fn select_physical_device(
    instance: &Arc<Instance>,
    device_extensions: DeviceExtensions,
) -> Arc<PhysicalDevice> {
    instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .min_by_key(|p| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap()
}

fn select_queue_family(physical_device: &Arc<PhysicalDevice>) -> u32 {
    // The Vulkan specs guarantee that a compliant implementation must provide at least one
    // queue that supports compute operations.
    physical_device
        .queue_family_properties()
        .iter()
        .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
        .unwrap() as u32
}

fn create_device(
    physical_device: &Arc<PhysicalDevice>,
    queue_family_index: u32,
    device_extensions: DeviceExtensions,
) -> (Arc<Device>, impl ExactSizeIterator<Item = Arc<Queue>>) {
    Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap()
}

fn create_memory_allocator(device: &Arc<Device>) -> Arc<StandardMemoryAllocator> {
    Arc::new(StandardMemoryAllocator::new_default(device.clone()))
}

fn create_descriptor_set_allocator(device: &Arc<Device>) -> Arc<StandardDescriptorSetAllocator> {
    Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ))
}

fn create_command_buffer_allocator(device: &Arc<Device>) -> Arc<StandardCommandBufferAllocator> {
    Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ))
}

fn create_compute_pipeline(
    device: &Arc<Device>,
    load_shader: fn(Arc<Device>) -> Result<Arc<ShaderModule>, Validated<VulkanError>>,
) -> Arc<ComputePipeline> {
    let cs = load_shader(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

fn create_descriptor_set(
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    pipeline: &Arc<ComputePipeline>,
    buffers: &[Subbuffer<impl ?Sized>],
) -> Arc<DescriptorSet> {
    DescriptorSet::new(
        descriptor_set_allocator.clone(),
        (&pipeline.layout().set_layouts()[0]).clone(),
        buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| WriteDescriptorSet::buffer(index as u32, buffer.clone())),
        [],
    )
    .unwrap()
}

fn create_command_buffer(
    command_buffer_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    pipeline: &Arc<ComputePipeline>,
    descriptor_set: &Arc<DescriptorSet>,
    push_constants: impl BufferContents,
    dispatch_group_counts: [u32; 3],
) -> Arc<PrimaryAutoCommandBuffer> {
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            descriptor_set.clone(),
        )
        .unwrap()
        .push_constants(pipeline.layout().clone(), 0, push_constants)
        .unwrap();

    // The command buffer only does one thing: execute the compute pipeline. This is called a
    // *dispatch* operation.
    unsafe { builder.dispatch(dispatch_group_counts) }.unwrap();

    // Finish building the command buffer by calling `build`.
    builder.build().unwrap()
}

fn execute(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
) {
    sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap()
}

fn report_aggregate(full_count: u32, interesting: &[bool]) {
    let total_count = TOTAL_BLOCK_COUNT;
    let interesting_count = interesting.iter().filter(|b| **b).count() as u32;
    let empty_count = total_count - full_count - interesting_count;

    println!("empty:       {empty_count:>5}");
    println!("full:        {full_count:>5}");
    println!("interesting: {interesting_count:>5}");
    println!("total:       {total_count:>5}");

    // region export as image
    let mut image_buffer = image::ImageBuffer::new(X_BLOCK_COUNT, Y_BLOCK_COUNT);
    let in_set_color = image::Rgb([0u8; 3]);
    let not_in_set_color = image::Rgb([255u8; 3]);
    for x in 0..X_BLOCK_COUNT {
        for y in 0..Y_BLOCK_COUNT {
            let index = y * X_BLOCK_COUNT + x;
            let color = if interesting[index as usize] {
                in_set_color
            } else {
                not_in_set_color
            };
            image_buffer.put_pixel(x, y, color);
        }
    }
    image_buffer.save("out2.png");
    // endregion
}

fn grow_interesting(old: &[bool], new: &mut [bool]) {
    let at = |x, y| {
        if x < X_BLOCK_COUNT && y < Y_BLOCK_COUNT {
            old[(y * X_BLOCK_COUNT + x) as usize]
        } else {
            false
        }
    };
    for x in 0..X_BLOCK_COUNT {
        for y in 0..Y_BLOCK_COUNT {
            let mut any = false;

            any |= at(x - 1, y - 1);
            any |= at(x - 1, y);
            any |= at(x - 1, y + 1);
            any |= at(x, y - 1);
            any |= at(x, y);
            any |= at(x, y + 1);
            any |= at(x + 1, y - 1);
            any |= at(x + 1, y);
            any |= at(x + 1, y + 1);

            new[(y * X_BLOCK_COUNT + x) as usize] = any;
        }
    }
}

#[derive(Default, Debug)]
struct AnyAllStats {
    /// not any and not all
    none: usize,
    /// any and all
    all: usize,
    /// all but not any
    invalid: usize,
    /// any but not all
    interesting: usize,
    total: usize,
}

fn any_all_stats(buffer: &[u32]) -> AnyAllStats {
    let mut stats = AnyAllStats::default();
    for &word in buffer {
        let any = word & 1 == 1;
        let all = word & 2 == 2;
        if !any && !all {
            stats.none += 1;
        }
        if any && all {
            stats.all += 1;
        }
        if any && !all {
            stats.interesting += 1;
        }
        if all && !any {
            stats.invalid += 1;
        }
        stats.total += 1;
    }
    stats
}

pub(crate) fn compress(mut data: &[bool]) -> Vec<i16> {
    let mut compressed = Vec::new();
    while let Some(next_true_index) = data.iter().position(|b| *b == true) {
        data = &data[next_true_index..];
        compressed.push(next_true_index as i16);

        loop {
            let (next_15, rest) = data.split_at(15.min(data.len()));
            if next_15.iter().all(|b| !*b) {
                break;
            }
            data = rest;

            let mut word = i16::MIN;
            for (i, &b) in next_15.iter().enumerate() {
                word |= (b as i16) << i;
            }
            compressed.push(word);
        }
    }
    compressed
}

pub(crate) fn decompress(mut compressed: &[i16], len: usize) -> Vec<bool> {
    let mut data = Vec::with_capacity(len);
    for &word in compressed {
        if word.is_negative() {
            for i in 0..15.min(len - data.len()) {
                let b = (word >> i) & 1 == 1;
                data.push(b);
            }
        } else {
            data.extend(repeat_n(false, word as usize));
        }
    }
    data.extend(repeat_n(false, len - data.len()));
    data
}

fn write_compressed(mut out: impl Write, compressed: &[i16]) -> io::Result<()> {
    write!(out, "&[")?;
    write!(out, "0x{:x}", compressed[0])?;
    for word in &compressed[1..] {
        write!(out, ", 0x{:x}", word)?;
    }
    write!(out, "]")?;
    Ok(())
}

fn write_results(mut out: impl Write, full_count: u32, full: &[bool], interesting: &[bool]) -> io::Result<()> {
    write!(out, "(")?;
    write!(out, "{full_count}")?;
    write!(out, ", ")?;
    write_compressed(&mut out, &compress(full))?;
    write!(out, ", ")?;
    write_compressed(&mut out, &compress(interesting))?;
    write!(out, ")")?;
    Ok(())
}

pub(crate) fn main() {
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };

    let library = load_library();
    let instance = create_instance(library);

    let physical_device = select_physical_device(&instance, device_extensions);
    let queue_family_index = select_queue_family(&physical_device);

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) =
        create_device(&physical_device, queue_family_index, device_extensions);

    let queue = queues.next().unwrap();

    let memory_allocator = create_memory_allocator(&device);
    let descriptor_set_allocator = create_descriptor_set_allocator(&device);
    let command_buffer_allocator = create_command_buffer_allocator(&device);

    let execute_mandelbrot = |data_buffer: Subbuffer<[u32]>,
                              mask_buffer: Subbuffer<[u32]>,
                              exp_min: f32,
                              exp_max: f32,
                              exp_steps: u32| {
        let pipeline = create_compute_pipeline(&device, mandelbrot::load);

        let descriptor_set = create_descriptor_set(
            &descriptor_set_allocator,
            &pipeline,
            &[data_buffer, mask_buffer],
        );

        let push_constants = mandelbrot::PushConstantData {
            x_resolution: X_RESOLUTION,
            y_resolution: Y_RESOLUTION,
            x_block_size: X_BLOCK_SIZE,
            y_block_size: Y_BLOCK_SIZE,
            x_block_count: X_BLOCK_COUNT,
            y_block_count: Y_BLOCK_COUNT,
            iters: ITERS,
            exp_min,
            exp_max,
            exp_steps,
        };

        let command_buffer = create_command_buffer(
            &command_buffer_allocator,
            &queue,
            &pipeline,
            &descriptor_set,
            push_constants,
            [X_RESOLUTION.div_ceil(8), Y_RESOLUTION.div_ceil(8), 1],
        );

        execute(&device, &queue, command_buffer);
    };

    let execute_aggregate = |pixel_buffer: Subbuffer<[u32]>,
                             block_buffer: Subbuffer<[u32]>,
                             mask_buffer: Subbuffer<[u32]>| {
        let pipeline = create_compute_pipeline(&device, aggregate::load);

        let descriptor_set = create_descriptor_set(
            &descriptor_set_allocator,
            &pipeline,
            &[pixel_buffer, block_buffer, mask_buffer],
        );

        let push_constants = aggregate::PushConstantData {
            x_resolution: X_RESOLUTION,
            y_resolution: Y_RESOLUTION,
            x_block_count: X_BLOCK_COUNT,
            y_block_count: Y_BLOCK_COUNT,
            x_block_size: X_BLOCK_SIZE,
            y_block_size: Y_BLOCK_SIZE,
        };

        let command_buffer = create_command_buffer(
            &command_buffer_allocator,
            &queue,
            &pipeline,
            &descriptor_set,
            push_constants,
            [X_BLOCK_COUNT.div_ceil(8), Y_BLOCK_COUNT.div_ceil(8), 1],
        );

        execute(&device, &queue, command_buffer);
    };

    let mandelbrot_count = |exp_min: f32, exp_max: f32, exp_steps: u32| {
        let data_buffer = allocate_u32_buffer_with(
            memory_allocator.clone(),
            repeat_n(2, TOTAL_RESOLUTION as usize),
        );
        let mask_buffer = allocate_u32_buffer_with(
            memory_allocator.clone(),
            repeat_n(1, TOTAL_BLOCK_COUNT as usize),
        );

        execute_mandelbrot(data_buffer.clone(), mask_buffer, exp_min, exp_max, exp_steps);

        let data_buffer_content = data_buffer.read().unwrap();
        let count_any = (0..TOTAL_RESOLUTION)
            .filter(|i| data_buffer_content[*i as usize] & 1 == 1)
            .count();
        let count_all = (0..TOTAL_RESOLUTION)
            .filter(|i| data_buffer_content[*i as usize] == 3)
            .count();
        let count_invalid = (0..TOTAL_RESOLUTION)
            .filter(|i| data_buffer_content[*i as usize] == 2)
            .count();

        // region export as image
        let mut image_buffer = image::ImageBuffer::new(X_RESOLUTION, Y_RESOLUTION);
        let in_set_color = image::Rgb([255u8; 3]);
        let some_in_set_color = image::Rgb([0u8; 3]);
        let not_in_set_color = image::Rgb([255u8; 3]);
        for x in 0..X_RESOLUTION {
            for y in 0..Y_RESOLUTION {
                let index = y * X_RESOLUTION + x;
                let any = data_buffer_content[index as usize] & 1 == 1;
                let all = data_buffer_content[index as usize] == 3;
                let color = if all {
                    in_set_color
                } else if any {
                    some_in_set_color
                } else {
                    not_in_set_color
                };
                image_buffer.put_pixel(x, y, color);
            }
        }
        image_buffer.save("out.png");
        // endregion

        (count_any, count_all)
    };

    let check_blocks = |exp_min: f32, exp_max: f32, exp_steps: u32, blocks: Subbuffer<[u32]>| {
        let data_buffer = allocate_u32_buffer_with(
            memory_allocator.clone(),
            repeat_n(2, TOTAL_RESOLUTION as usize),
        );
        let mask_buffer = allocate_u32_buffer_with(
            memory_allocator.clone(),
            repeat_n(1, TOTAL_BLOCK_COUNT as usize),
        );

        execute_mandelbrot(data_buffer.clone(), mask_buffer, exp_min, exp_max, exp_steps);

        let block_buffer_content = blocks.read().unwrap();
        let data_buffer_content = data_buffer.read().unwrap();
        for x in 0..X_RESOLUTION {
            for y in 0..Y_RESOLUTION {
                let index = y * X_RESOLUTION + x;
                let pixel_content = data_buffer_content[index as usize];
                let pixel_any = pixel_content & 1 == 1;
                let pixel_all = pixel_content == 3;

                let block_y = y / Y_BLOCK_SIZE;
                let block_x = x / X_BLOCK_SIZE;
                let block_index = block_y * X_BLOCK_COUNT + block_x;
                let block_content = block_buffer_content[block_index as usize];
                let block_any = block_content & 1 == 1;

                if pixel_any {
                    assert!(block_any, "{x} {y}");
                }
            }
        }
    };

    let aggregate_range = |min: f32, max: f32, steps: u32| -> (u32, Vec<bool>, Vec<bool>) {
        let pixel_buffer = allocate_u32_buffer_with(
            memory_allocator.clone(),
            repeat_n(2, TOTAL_RESOLUTION as usize),
        );
        let block_buffer = allocate_u32_buffer_with(
            memory_allocator.clone(),
            repeat_n(2, TOTAL_BLOCK_COUNT as usize),
        );
        let mask_buffer = allocate_u32_buffer_with(
            memory_allocator.clone(),
            repeat_n(1, TOTAL_BLOCK_COUNT as usize),
        );

        const INITIAL_STEPS: u32 = 100;

        execute_mandelbrot(
            pixel_buffer.clone(),
            mask_buffer.clone(),
            min,
            max,
            INITIAL_STEPS,
        );
        execute_aggregate(
            pixel_buffer.clone(),
            block_buffer.clone(),
            mask_buffer.clone(),
        );

        let block_buffer_content = block_buffer.read().unwrap();

        // dbg!(any_all_stats(&*block_buffer_content));

        let initial_full_count = block_buffer_content.iter().filter(|x| **x == 3).count() as u32;
        let initial_interesting: Vec<bool> = block_buffer_content.iter().map(|x| *x == 1).collect();

        let mut interesting_grow = vec![false; TOTAL_BLOCK_COUNT as usize];

        grow_interesting(&initial_interesting, &mut interesting_grow);

        drop(block_buffer_content);

        let mask_buffer = allocate_u32_buffer_with(
            memory_allocator.clone(),
            interesting_grow
                .iter()
                .map(|interesting| *interesting as u32),
        );

        execute_mandelbrot(
            pixel_buffer.clone(),
            mask_buffer.clone(),
            min,
            max,
            steps,
        );
        execute_aggregate(
            pixel_buffer.clone(),
            block_buffer.clone(),
            mask_buffer.clone(),
        );

        // check_blocks(min, max, 10_000, block_buffer.clone());

        let block_buffer_content = block_buffer.read().unwrap();

        dbg!(any_all_stats(&*block_buffer_content));

        let full_count = block_buffer_content.iter().filter(|x| **x == 3).count() as u32;
        let interesting: Vec<bool> = block_buffer_content.iter().map(|x| *x == 1).collect();
        let full: Vec<bool> = block_buffer_content.iter().map(|x| *x == 3).collect();

        for i in 0..TOTAL_BLOCK_COUNT as usize {
            if interesting[i] {
                assert!(interesting_grow[i]);
            }
        }

        // region export as image
        /*let mut image_buffer = image::ImageBuffer::new(X_BLOCK_COUNT, Y_BLOCK_COUNT);
        let in_set_color = image::Rgb([0u8; 3]);
        let some_in_set_color = image::Rgb([255u8, 0, 0]);
        let not_in_set_color = image::Rgb([255u8; 3]);
        for x in 0..X_BLOCK_COUNT {
            for y in 0..Y_BLOCK_COUNT {
                let index = y * X_BLOCK_COUNT + x;
                let any = block_buffer_content[index as usize] & 1 == 1;
                let all = block_buffer_content[index as usize] == 3;
                let color = if all {
                    in_set_color
                } else if any {
                    some_in_set_color
                } else {
                    not_in_set_color
                };
                image_buffer.put_pixel(x, y, color);
            }
        }
        image_buffer.save("out.png");*/
        // endregion

        (full_count, full, interesting)
    };

    /*let start = Instant::now();

    let min = 2.0;
    // for 2.001:     17544 interesting
    // for 2.0001:     9419 interesting
    // for 2.00001:    4483 interesting
    // for 2.000001:   1542 interesting
    // for 2.0000001:     0 interesting
    let max = 2.0001;
    let steps = 1_000;
    let (count_any, count_all) = mandelbrot_count(min, max, steps);

    println!("count_any:   {count_any}");
    println!("count_all:   {count_all}");
    println!("interesting: {}", count_any - count_all);

    println!("took {:?}", start.elapsed());*/

    let start = Instant::now();

    // for  1_000 steps: 1168
    // for  5_000 steps: 1196
    // for 10_000 steps: 1196
    // for 20_000 steps: 1196
    // let (full_count, full, interesting) = aggregate_range(2.5, 2.501, 10_000);
    // println!("compressed size: {} bytes", compress(&interesting).len() * 2);
    // let reconstructed = decompress(&compress(&interesting), interesting.len());
    // assert_eq!(interesting, reconstructed);
    // println!("took {:?}", start.elapsed());
    //
    // report_aggregate(full_count, &interesting);

    let file = OpenOptions::new().create(true).append(true).open("data.txt").unwrap();
    let mut out = BufWriter::new(file);
    // let mut out = io::stdout();
    let start = Instant::now();

    writeln!(out, "[").unwrap();
    for i in 0..1000 {
        write!(out, "    ").unwrap();
        let min = 2.0 + i as f32 / 1000.0;
        let max = min + 1.0 / 1000.0;
        let (full_count, full, interesting) = aggregate_range(min, max, 1_000);
        write_results(&mut out, full_count, &full, &interesting).unwrap();
        writeln!(out, ", ").unwrap();

        out.flush().unwrap();

        println!("i = {i:>3}, dt = {:?}", start.elapsed());
    }
    writeln!(out, "]").unwrap();
    out.flush().unwrap();
}
