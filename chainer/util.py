from chainer.links.caffe.protobuf3 import caffe_pb2 as caffe_pb
from chainer.function_hooks import timer
from chainer.links.caffe import CaffeFunction
from chainer import cuda
from chainer import function
import chainer.links as L
import chainer.functions as F
from chainer import link
from chainer import function

import collections
import threading
import time
import sys
import numpy


class TimerHook(function.FunctionHook):
	"""Function hook for measuring elapsed time of functions.

	Attributes:
		call_history: List of measurement results. It consists of pairs of
			the function that calls this hook and the elapsed time
			the function consumes.
	"""

	name = 'TimerHook'

	def __init__(self):
		self.call_history = []
		self.layer_name = ''

	def _preprocess(self):
		if self.xp == numpy:
			self.start = time.time()
		else:
			self.start = cuda.Event()
			self.stop = cuda.Event()
			self.start.record()

	def forward_preprocess(self, function, in_data):
		self.xp = cuda.get_array_module(*in_data)
		self._preprocess()

	def backward_preprocess(self, function, in_data, out_grad):
		self.xp = cuda.get_array_module(*(in_data + out_grad))
		self._preprocess()

	def _postprocess(self, function):
		if self.xp == numpy:
			self.stop = time.time()
			elapsed_time = self.stop - self.start
		else:
			self.stop.record()
			self.stop.synchronize()
			# Note that `get_elapsed_time` returns result in milliseconds
			elapsed_time = cuda.cupy.cuda.get_elapsed_time(
				self.start, self.stop) / 1000
		if self.layer_name is None:
			return
		self.call_history.append((self.layer_name, elapsed_time))

	def forward_postprocess(self, function, in_data):
		xp = cuda.get_array_module(*in_data)
		assert xp == self.xp
		self._postprocess(function)

	def backward_postprocess(self, function, in_data, out_grad):
		xp = cuda.get_array_module(*(in_data + out_grad))
		assert xp == self.xp
		self._postprocess(function)

	def total_time(self):
		"""Returns total elapsed time in seconds."""
		return sum(t for (_, t) in self.call_history)

	def print_layer_time(self):
		layer_time_dict = {}
		for func, time in self.call_history:
		# func_name = func.label
			func_name = func
			if layer_time_dict.get(func_name) is None:
				layer_time_dict[func_name] = {'time': time, 'number': 1}
			else:
				layer_time_dict[func_name]['time'] += time
				layer_time_dict[func_name]['number'] += 1

		print '================================================'
		keys = layer_time_dict.keys()
		keys.sort()
		for name in keys:
			record = layer_time_dict[name]
			print '[{}]:'.format(name)
			print 'total time: {} ms'.format(record['time'] * 1000)
			print 'average time: {} ms\n'.format(float(record['time']) * 1000/ record['number'])
		print '================================================'

origin_call = {}
def ModelWrapper(model, timer_hook):
	cls = model.__class__
	class wrapper_cls(cls):
		def __init__(self):
			attr_before = list(self.__dict__)
			super(wrapper_cls, self).__init__()
			attr_after = list(self.__dict__)

			for attr in attr_after:
				# attr is new added layer
				if isinstance(self[attr], link.Link) or isinstance(self[attr], function.Function):
					def wrapper_function(this, x):
						timer_hook.layer_name = this._layer_name
						with timer_hook:
							ret = origin_call[this.__class__](this, x)
						timer_hook.layer_name = None
						return ret

					self[attr]._layer_name = attr
					if self[attr].__class__ not in origin_call:
						origin_call.update({self[attr].__class__: self[attr].__class__.__call__})
						self[attr].__class__.__call__ = wrapper_function

	return wrapper_cls()
