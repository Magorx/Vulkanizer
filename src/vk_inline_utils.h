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
