"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""
from torch import nn

class SqueezeLayer(nn.Module):
	def __init__(self, factor, level, name='squeeze'):
		super(SqueezeLayer, self).__init__()
		self.factor = factor
		self.name = name
		self.level = level

	def _inverse(self, z, **kwargs):
		output = unsqueeze2d(z, self.factor)
		return output

	def _forward_and_log_det_jacobian(self, x, **kwargs):
		output = squeeze2d(x, self.factor)
		return output, 0

def squeeze2d(input, factor=2):
	#assert factor >= 1 and isinstance(factor, int)
	if factor == 1:
		return input
	size = input.size()
	B = size[0]
	C = size[1]
	H = size[2]
	W = size[3]
	assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
	x = input.view(B, C, H // factor, factor, W // factor, factor)
	x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
	x = x.view(B, C * factor * factor, H // factor, W // factor)
	return x


def unsqueeze2d(input, factor=2):
	assert factor >= 1 and isinstance(factor, int)
	factor2 = factor ** 2
	if factor == 1:
		return input
	size = input.size()
	B = size[0]
	C = size[1]
	H = size[2]
	W = size[3]
	assert C % (factor2) == 0, "{}".format(C)
	x = input.view(B, C // factor2, factor, factor, H, W)
	x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
	x = x.view(B, C // (factor2), H * factor, W * factor)
	return x

