import torch
import torch.nn.functional as F

def diagonal_loss(distance_matrix):
    return F.mse_loss(
        torch.diagonal(distance_matrix),
        torch.zeros_like(torch.diagonal(distance_matrix)),
    )

def symmetry_loss(distance_matrix):
    return F.mse_loss(distance_matrix, distance_matrix.transpose(-2, -1))

def non_negative_loss(distance_matrix):
    # This function penalizes negative values in the distance matrix
    negative_values = torch.relu(
        -distance_matrix
    )  # ReLU gives us just the negative values
    return negative_values.mean()  # Return the average negative value

def triangle_inequality_loss(distance_matrix, n=1000):
    # Get the shape of the last two dimensions
    n, m = distance_matrix.shape[-2:]

    # Create a 2D grid of indices for each dimension
    i, j = torch.meshgrid(torch.arange(n), torch.arange(m))

    # Create a 1D array of indices for the last two dimensions
    indices = torch.arange(n * m)
    indices = indices[torch.randperm(indices.size()[0])]
    indices = indices[:n]
    # Get all combinations of three indices
    combinations = torch.combinations(indices, 3)

    # Convert 1D indices back to 2D
    i, j, k = combinations.unbind(1)
    i1, i2 = i // m, i % m
    j1, j2 = j // m, j % m
    k1, k2 = k // m, k % m

    # Calculate the violation of the triangle inequality
    violation = (
        distance_matrix[..., i1, i2]
        + distance_matrix[..., j1, j2]
        - distance_matrix[..., k1, k2]
    )

    # Return the mean of the rectified violation
    return torch.relu(violation).nanmean()

def clockwise_order_loss(distance_matrix):
    # Get the size of the last two dimensions
    n, m = distance_matrix.shape[-2:]

    # Create expected order tensor
    expected_order = torch.arange(n * m).reshape(1, 1, n, m).to(distance_matrix.device)

    # Create expected difference tensor
    expected_diff = (
        expected_order.unsqueeze(-1).unsqueeze(-1)
        - expected_order.unsqueeze(-3).unsqueeze(-3)
    ) % (n * m)

    # Expand dimensions for broadcasting
    distance_matrix_exp = distance_matrix.unsqueeze(-2).unsqueeze(-2)

    # Create observed difference tensor
    observed_diff = (distance_matrix_exp > distance_matrix_exp.transpose(-1, -2)).long()

    # Ensure expected_diff and observed_diff have the same shape
    expected_diff = expected_diff.expand_as(observed_diff)

    # Calculate loss
    loss = (observed_diff != expected_diff).float().mean()

    return loss

losses = { 'diagonal': diagonal_loss
         , 'symmetry': symmetry_loss
         , 'non_negative': non_negative_loss
         , 'triangle_inequality': triangle_inequality_loss
         }
