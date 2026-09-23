import onnxruntime as ort
import numpy as np
# from isaaclab.observations import gravity_body_frame

class ValueProcessor:

    def __init__(self, onnx_path, map_out = False):
        self.model = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

        # Verify it's actually using GPU
        print("Execution providers:", self.model.get_providers())
        print("Running on:", self.model.get_provider_options())
        
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape[1]
        self.output_shape = self.model.get_outputs()[0].shape
        self.map_out = map_out

    def inference(self, *args):
        if self.map_out:
            return self.inference_map(*args)
        return self.inference_goal(*args)

    # #ROUGH
    # def inference_map(self, elevation_map):
    #     inputs = {self.input_name: elevation_map}
    #     model_output = self.model.run(None, inputs)[0]
    #     N, M, _ = model_output.shape
    #     output = np.zeros((N, M), dtype=np.float32)
    #     output[:10,:] = model_output[:10, :, 0]
    #     output[10:,:] = model_output[10:, :, 7]
    #     return output

    #FLAT and HOUND
    def inference_map(self, elevation_map):
        inputs = {self.input_name: elevation_map}
        model_output = self.model.run(None, inputs)[0]
        N, M, _ = model_output.shape
        output = model_output[0, :, :]
        return output

    # def inference_goal(self, goals, orientation, elevation_map):
    #     gravity = gravity_body_frame(orientation)
    #     input_data = np.zeros((goals.shape[0], self.input_shape), dtype=np.float32)
    #     input_data[:, :self.input_shape - 5] = elevation_map
    #     input_data[:, self.input_shape - 5:self.input_shape - 2] = gravity
    #     input_data[:, self.input_shape - 2:] = goals
    #     inputs = {self.input_name: input_data}
    #     model_output = self.model.run(None, inputs)[0]
    #     return model_output