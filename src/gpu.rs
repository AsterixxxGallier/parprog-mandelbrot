// This example demonstrates how to use the compute capabilities of Vulkan.
//
// While graphics cards have traditionally been used for graphical operations, over time they have
// been more or more used for general-purpose operations as well. This is called "General-Purpose
// GPU", or *GPGPU*. This is what this example demonstrates.

use crate::{ITERS, TOTAL_RESOLUTION, X_RESOLUTION, Y_RESOLUTION};
use image::Rgb;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{
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
    VulkanLibrary,
};

mod cs {
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

                float re = (float(x) / float(pc.x_resolution)) * 2.0 - 1.0;
                float im = (float(y) / float(pc.y_resolution)) * 2.0 - 1.0;

                data[index] = uint(mandelbrot(re, im, pc.exp, pc.iters));
            }
        ",
    }
}

pub(crate) fn main() {
    // As with other examples, the first step is to create an instance.
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    // Choose which physical device to use.
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one
            // queue that supports compute operations.
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    // Now initializing the device.
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retrieve the first and only element of the
    // iterator and throw it away.
    let queue = queues.next().unwrap();

    // Now let's get to the actual example.
    //
    // What we are going to do is very basic: we are going to fill a buffer with 64k integers and
    // ask the GPU to multiply each of them by 12.
    //
    // GPUs are very good at parallel computations (SIMD-like operations), and thus will do this
    // much more quickly than a CPU would do. While a CPU would typically multiply them one by one
    // or four by four, a GPU will do it by groups of 32 or 64.
    //
    // Note however that in a real-life situation for such a simple operation the cost of accessing
    // memory usually outweighs the benefits of a faster calculation. Since both the CPU and the
    // GPU will need to access data, there is no other choice but to transfer the data through the
    // slow PCI express bus.

    // We need to create the compute pipeline that describes our operation.
    //
    // If you are familiar with graphics pipeline, the principle is the same except that compute
    // pipelines are much simpler to create.
    let pipeline = {
        let cs = cs::load(device.clone())
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
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    // We start by creating the buffer that will store the data.
    let data_buffer = Buffer::new_unsized::<[u32]>(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        TOTAL_RESOLUTION as DeviceSize,
    )
    .unwrap();

    // In order to let the shader access the buffer, we need to build a *descriptor set* that
    // contains the buffer.
    //
    // The resources that we bind to the descriptor set must match the resources expected by the
    // pipeline which we pass as the first parameter.
    //
    // If you want to run the pipeline on multiple different buffers, you need to create multiple
    // descriptor sets that each contain the buffer you want to run the shader on.
    let layout = &pipeline.layout().set_layouts()[0];
    let set = DescriptorSet::new(
        descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
        [],
    )
    .unwrap();

    // The `vulkano_shaders::shaders!` macro generates a struct with the correct representation of
    // the push constants struct specified in the shader. Here we create an instance of the
    // generated struct.
    let push_constants = cs::PushConstantData {
        x_resolution: X_RESOLUTION,
        y_resolution: Y_RESOLUTION,
        iters: ITERS,
        exp: 2.5,
    };

    // In order to execute our operation, we have to build a command buffer.
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    // Note that we clone the pipeline and the set. Since they are both wrapped in an `Arc`,
    // this only clones the `Arc` and not the whole pipeline or set (which aren't cloneable
    // anyway). In this example we would avoid cloning them since this is the last time we use
    // them, but in real code you would probably need to clone them.
    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap()
        .push_constants(pipeline.layout().clone(), 0, push_constants)
        .unwrap();

    // The command buffer only does one thing: execute the compute pipeline. This is called a
    // *dispatch* operation.
    unsafe { builder.dispatch([X_RESOLUTION.div_ceil(8), Y_RESOLUTION.div_ceil(8), 1]) }.unwrap();

    // Finish building the command buffer by calling `build`.
    let command_buffer = builder.build().unwrap();

    // Let's execute this command buffer now.
    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        // This line instructs the GPU to signal a *fence* once the command buffer has finished
        // execution. A fence is a Vulkan object that allows the CPU to know when the GPU has
        // reached a certain point. We need to signal a fence here because below we want to block
        // the CPU until the GPU has reached that point in the execution.
        .then_signal_fence_and_flush()
        .unwrap();

    // Blocks execution until the GPU has finished the operation. This method only exists on the
    // future that corresponds to a signalled fence. In other words, this method wouldn't be
    // available if we didn't call `.then_signal_fence_and_flush()` earlier. The `None` parameter
    // is an optional timeout.
    //
    // Note however that dropping the `future` variable (with `drop(future)` for example) would
    // block execution as well, and this would be the case even if we didn't call
    // `.then_signal_fence_and_flush()`. Therefore the actual point of calling
    // `.then_signal_fence_and_flush()` and `.wait()` is to make things more explicit. In the
    // future, if the Rust language gets linear types vulkano may get modified so that only
    // fence-signalled futures can get destroyed like this.
    future.wait(None).unwrap();

    // Now that the GPU is done, the content of the buffer should have been modified. Let's check
    // it out. The call to `read()` would return an error if the buffer was still in use by the
    // GPU.
    let data_buffer_content = data_buffer.read().unwrap();
    let count = (0..TOTAL_RESOLUTION)
        .filter(|i| data_buffer_content[*i as usize] != 0)
        .count();
    // for n in 0..65536u32 {
    //     let expected = (n as f32) * 12.0 + 1.0;
    //     let actual = data_buffer_content[n as usize];
    //     let error = ((actual - expected) / expected.max(f32::EPSILON)).abs();
    //     assert!(error < 1e-5, "expected {expected}, got {actual}");
    // }

    let mut image_buffer = image::ImageBuffer::new(X_RESOLUTION, Y_RESOLUTION);
    let in_set_color = Rgb([0u8; 3]);
    let not_in_set_color = Rgb([255u8; 3]);
    for x in 0..X_RESOLUTION {
        for y in 0..Y_RESOLUTION {
            let index = y * X_RESOLUTION + x;
            let in_set = data_buffer_content[index as usize] != 0;
            let color = if in_set { in_set_color } else { not_in_set_color };
            image_buffer.put_pixel(x, y, color);
        }
    }
    image_buffer.save("out.png");

    println!("count: {count}");
    println!("Success");
}
