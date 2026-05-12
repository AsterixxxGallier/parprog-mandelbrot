use std::iter::repeat_n;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::memory::allocator::{AllocationCreateInfo, GenericMemoryAllocatorCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::{instance::*, sync, VulkanLibrary};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::{StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo};
use vulkano::descriptor_set::{DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateFlags, PipelineLayoutCreateInfo};
use vulkano::sync::GpuFuture;
use crate::{TOTAL_RESOLUTION, X_RESOLUTION, Y_RESOLUTION};

fn load_library() -> Arc<VulkanLibrary> {
    // SAFETY: We can't really make sure that the loaded library is valid. Here's hoping.
    let result = unsafe { VulkanLibrary::new() };
    result.expect("no local Vulkan library/DLL")
}

fn create_instance(library: &Arc<VulkanLibrary>) -> Arc<Instance> {
    Instance::new(
        &library,
        &InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
        .expect("failed to create instance")
}

fn select_physical_device(instance: &Arc<Instance>) -> Arc<PhysicalDevice> {
    instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .min_by_key(|device| match device.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no devices available")
}

fn select_queue_family(physical_device: &Arc<PhysicalDevice>) -> u32 {
    physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family| queue_family.queue_flags.contains(QueueFlags::COMPUTE))
        .expect("couldn't find queue family with compute capability") as u32
}

fn create_device(
    physical_device: &Arc<PhysicalDevice>,
    queue_family_index: u32,
) -> (Arc<Device>, impl ExactSizeIterator<Item=Arc<Queue>>) {
    Device::new(
        physical_device,
        &DeviceCreateInfo {
            // here we pass the desired queue family to use by index
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
        .expect("failed to create device")
}

fn create_allocator(device: &Arc<Device>) -> Arc<StandardMemoryAllocator> {
    Arc::new(StandardMemoryAllocator::new(
        device,
        &GenericMemoryAllocatorCreateInfo::default(),
    ))
}

fn create_buffer(allocator: &Arc<StandardMemoryAllocator>) -> Subbuffer<[u32]> {
    Buffer::from_iter(
        allocator,
        &BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        repeat_n(1u32, TOTAL_RESOLUTION),
    ).expect("failed to create buffer")
}

mod compute_shader {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                // needs to be the same values as X_RESOLUTION and Y_RESOLUTION
                if (gl_GlobalInvocationID.x >= 1923 || gl_GlobalInvocationID.y >= 1447) {
                    return;
                }
                uint idx = gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x + gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }
        ",
    }
}

pub(crate) fn main() {
    let library = load_library();
    let instance = create_instance(&library);
    let physical_device = select_physical_device(&instance);
    let queue_family_index = select_queue_family(&physical_device);
    let (device, mut queues) = create_device(&physical_device, queue_family_index);
    let queue = queues.next().expect("no queue in selected queue family");
    let allocator = create_allocator(&device);

    let buffer = create_buffer(&allocator);
    let shader = compute_shader::load(&device).expect("failed to load compute shader");
    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(&cs);
    let layout = PipelineLayout::new(
        &device,
        &PipelineLayoutCreateInfo {
            flags: PipelineLayoutCreateFlags::empty(),
            set_layouts: &[&DescriptorSetLayout::new(&device, &DescriptorSetLayoutCreateInfo {
                flags: DescriptorSetLayoutCreateFlags::empty(),
                bindings: &[DescriptorSetLayoutBinding::new(DescriptorType::StorageBuffer)],
                ..Default::default()
            }).expect("could not create descriptor set layout")],
            push_constant_ranges: &[],
            ..Default::default()
        },
    )
        .unwrap();

    let compute_pipeline = ComputePipeline::new(
        &device,
        None,
        &ComputePipelineCreateInfo::new(stage, &layout),
    )
        .expect("failed to create compute pipeline");

    let descriptor_set_allocator =
        Arc::new(StandardDescriptorSetAllocator::new(&device, &StandardDescriptorSetAllocatorCreateInfo::default()));
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = DescriptorSet::new(
        &descriptor_set_allocator,
        &descriptor_set_layout,
        &[WriteDescriptorSet::buffer(0, &DescriptorBufferInfo {
            buffer: Some(buffer.buffer()),
            offset: buffer.offset(),
            range: Some(buffer.size()),
        })],
        &[],
    )
        .unwrap();

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        &device,
        &StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
        .unwrap();

    let work_group_counts = [X_RESOLUTION as u32 / 8, Y_RESOLUTION as u32 / 8, 1];

    unsafe {
        command_buffer_builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )
            .unwrap()
            .dispatch(work_group_counts)
            .unwrap();
    }

    let command_buffer = command_buffer_builder.build().unwrap();
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let content = buffer.read().unwrap();
    for &val in content.iter() {
        assert_eq!(val, 12);
    }

    println!("Everything succeeded!");
}
