// This example demonstrates how to use the compute capabilities of Vulkan.
//
// While graphics cards have traditionally been used for graphical operations, over time they have
// been more or more used for general-purpose operations as well. This is called "General-Purpose
// GPU", or *GPGPU*. This is what this example demonstrates.

use crate::{ITERS, TOTAL_RESOLUTION, X_RESOLUTION, Y_RESOLUTION};
use std::sync::Arc;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{CommandBuffer, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::Queue;
use vulkano::shader::ShaderModule;
use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage}, command_buffer, command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
}, descriptor_set::{
    allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
}, device::{
    physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    QueueFlags,
}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
    compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline,
    PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
}, sync::{self, GpuFuture}, DeviceSize, Validated, VulkanError, VulkanLibrary};

mod mandelbrot {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            };

            layout(push_constant) uniform PushConstantData {
                uint x_resolution;
                uint y_resolution;
                uint iters;
                float exp;
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

                uint index = y * pc.x_resolution + x;

                float re = (float(x) / float(pc.x_resolution)) * 4.0 - 2.0;
                float im = (float(y) / float(pc.y_resolution)) * 4.0 - 2.0;

                data[index] = uint(mandelbrot(re, im, pc.exp, pc.iters));
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

            layout(set = 0, binding = 0) buffer InSet {
                uint in_set[1923 * 1447];
            };

            layout(set = 0, binding = 1) buffer NotEmpty {
                uint not_empty[(1923 / 8) * (1447 / 8)];
            };

            layout(set = 0, binding = 2) buffer NotFull {
                uint not_full[(1923 / 8) * (1447 / 8)];
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

                uint min_pixel_x = block_x * pc.x_block_size;
                uint min_pixel_y = block_y * pc.y_block_size;

                uint max_pixel_x = min(int(pc.x_resolution), int(min_pixel_x + pc.x_block_size));
                uint max_pixel_y = min(int(pc.y_resolution), int(min_pixel_y + pc.y_block_size));

                bool block_not_empty = false;
                bool block_not_full = false;

                for (uint pixel_x = min_pixel_x; pixel_x < max_pixel_x; pixel_x++) {
                    for (uint pixel_y = min_pixel_y; pixel_y < max_pixel_y; pixel_y++) {
                        uint pixel_index = pixel_y * pc.x_resolution + pixel_x;
                        bool pixel_in_set = bool(in_set[pixel_index]);
                        if (pixel_in_set) block_not_empty = true;
                        else block_not_full = true;
                    }
                }

                not_empty[block_index] |= uint(block_not_empty);
                not_full[block_index] |= uint(block_not_full);
            }
        ",
    }
}

// keep in sync with shader
const X_BLOCK_SIZE: u32 = 8;
const Y_BLOCK_SIZE: u32 = 8;

const X_BLOCK_COUNT: u32 = X_RESOLUTION.div_ceil(X_BLOCK_SIZE);
const Y_BLOCK_COUNT: u32 = Y_RESOLUTION.div_ceil(Y_BLOCK_SIZE);
const TOTAL_BLOCK_COUNT: u32 = X_BLOCK_COUNT * Y_BLOCK_COUNT;

fn allocate_bool_buffer(
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

fn execute(device: &Arc<Device>, queue: &Arc<Queue>, command_buffer: Arc<PrimaryAutoCommandBuffer>) {
    sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap()
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

    let execute_mandelbrot = |data_buffer: Subbuffer<[u32]>, exp: f32| {
        let pipeline = create_compute_pipeline(&device, mandelbrot::load);

        let descriptor_set =
            create_descriptor_set(&descriptor_set_allocator, &pipeline, &[data_buffer]);

        let push_constants = mandelbrot::PushConstantData {
            x_resolution: X_RESOLUTION,
            y_resolution: Y_RESOLUTION,
            iters: ITERS,
            exp,
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

    let execute_aggregate =
        |mandelbrot_buffer: Subbuffer<[u32]>,
         not_empty_buffer: Subbuffer<[u32]>,
         not_full_buffer: Subbuffer<[u32]>| {
            let pipeline = create_compute_pipeline(&device, aggregate::load);

            let descriptor_set = create_descriptor_set(
                &descriptor_set_allocator,
                &pipeline,
                &[mandelbrot_buffer, not_empty_buffer, not_full_buffer],
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

    let mandelbrot_count = |exp: f32| {
        let data_buffer =
            allocate_bool_buffer(memory_allocator.clone(), TOTAL_RESOLUTION as DeviceSize);

        execute_mandelbrot(data_buffer.clone(), exp);

        let data_buffer_content = data_buffer.read().unwrap();
        let count = (0..TOTAL_RESOLUTION)
            .filter(|i| data_buffer_content[*i as usize] != 0)
            .count();

        // region export as image
        let mut image_buffer = image::ImageBuffer::new(X_RESOLUTION, Y_RESOLUTION);
        let in_set_color = image::Rgb([0u8; 3]);
        let not_in_set_color = image::Rgb([255u8; 3]);
        for x in 0..X_RESOLUTION {
            for y in 0..Y_RESOLUTION {
                let index = y * X_RESOLUTION + x;
                let in_set = data_buffer_content[index as usize] != 0;
                let color = if in_set {
                    in_set_color
                } else {
                    not_in_set_color
                };
                image_buffer.put_pixel(x, y, color);
            }
        }
        image_buffer.save("out.png");
        // endregion

        count
    };

    let mandelbrot_aggregate = |mandelbrot_buffer| {
        let not_empty_buffer =
            allocate_bool_buffer(memory_allocator.clone(), TOTAL_BLOCK_COUNT as DeviceSize);
        let not_full_buffer =
            allocate_bool_buffer(memory_allocator.clone(), TOTAL_BLOCK_COUNT as DeviceSize);

        let command_buffer = execute_aggregate(
            mandelbrot_buffer,
            not_empty_buffer.clone(),
            not_full_buffer.clone(),
        );

        let not_empty_buffer_content = not_empty_buffer.read().unwrap();
        let not_full_buffer_content = not_full_buffer.read().unwrap();

        println!(
            "not empty count: {}",
            not_empty_buffer_content.iter().filter(|x| **x != 0).count()
        );
        println!(
            "not full count:  {}",
            not_full_buffer_content.iter().filter(|x| **x != 0).count()
        );
        println!(
            "interesting count:  {}",
            not_empty_buffer_content
                .iter()
                .zip(not_full_buffer_content.iter())
                .filter(|(x, y)| **x != 0 && **y != 0)
                .count()
        );
        println!("total count:  {}", TOTAL_BLOCK_COUNT,);

        // region export as image
        let mut image_buffer = image::ImageBuffer::new(X_BLOCK_COUNT, Y_BLOCK_COUNT);
        let in_set_color = image::Rgb([0u8; 3]);
        let not_in_set_color = image::Rgb([255u8; 3]);
        for x in 0..X_BLOCK_COUNT {
            for y in 0..Y_BLOCK_COUNT {
                let index = y * X_BLOCK_COUNT + x;
                let not_empty = not_empty_buffer_content[index as usize] != 0;
                let not_full = not_full_buffer_content[index as usize] != 0;
                let color = if not_empty && not_full {
                    in_set_color
                } else {
                    not_in_set_color
                };
                image_buffer.put_pixel(x, y, color);
            }
        }
        image_buffer.save("out2.png");
        // endregion
    };

    let mandelbrot_then_aggregate = |exp: f32| {
        // We start by creating the buffer that will store the data.
        let data_buffer = allocate_bool_buffer(memory_allocator.clone(), TOTAL_RESOLUTION as DeviceSize);

        execute_mandelbrot(data_buffer.clone(), exp);

        mandelbrot_aggregate(data_buffer);
    };

    let count = mandelbrot_count(2.5);

    mandelbrot_then_aggregate(2.5);

    println!("count: {count}");
    println!("Success");
}
