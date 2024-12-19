import torch
import coremltools as ct
from model.waveunet import Waveunet

def convert():
    target_outputs = int(2 * 12000)
    num_features = [32 * 2 ** i for i in range(0, 6)]
    model = Waveunet(num_features, kernel_size=5, target_output_size=target_outputs, strides=4)
    # Load pretrained weights if available
    # checkpoint_path = "path_to_checkpoint.pth"
    # model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    print(model.shapes)

    # Create example input matching your model's input size
    example_input = torch.rand(1, 2, model.shapes["input_frames"])  # Adjust input shape as needed (e.g., batch_size, channels, samples)

    # Trace the model
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=example_input.shape)],
    )

    # Save CoreML model
    mlmodel.save("waveunet.mlpackage")
    print("Model successfully converted to CoreML!")


def main():
    convert()


if __name__ == '__main__':
    main()