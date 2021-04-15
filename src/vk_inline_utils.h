inline VkResult KVK_createDescriptorSetLayout(
	const VkDevice device,
	const int binding,
	const VkDescriptorType descriptorType,
	// const uint32_t descriptorCount,
	const VkShaderStageFlags stageFlags,
	//const VkSampler *pImmutableSamplers,
	VkDescriptorSetLayout &descriptorSetLayout
	)
{
	VkDescriptorSetLayoutBinding layoutBinding {};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = descriptorType;
    layoutBinding.descriptorCount = 1;

    layoutBinding.stageFlags = stageFlags;
    layoutBinding.pImmutableSamplers = nullptr; // Optional

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &layoutBinding;

    return vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
}

// template<typename T>
// inline VkResult KVK_createStagingBufferAndUpload(
// 	const VkDevice device,
// 	std::vector<T> dataBuffer,
// 	VkBuffer &holder,
// 	VkDeviceMemory &holderMemory
// 	)
// {
// 	VkDeviceSize bufferSize = sizeof(T) * dataBuffer.size();

//     VkBuffer stagingBuffer;
//     VkDeviceMemory stagingBufferMemory;
//     createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

//     void* data;
//     vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
//     memcpy(data, dataBuffer.data(), (size_t) bufferSize);
//     vkUnmapMemory(device, stagingBufferMemory);

//     createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, holder, holderMemory);

//     copyBuffer(stagingBuffer, holder, bufferSize);

//     vkDestroyBuffer(device, stagingBuffer, nullptr);
//     vkFreeMemory(device, stagingBufferMemory, nullptr);
// }
