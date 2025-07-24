# main.tf - Terraform configuration for GCP GPU Instance

# Configure the Google Cloud provider
provider "google" {
  project = "hpe-managed-services" # <-- IMPORTANT: Replace with your GCP Project ID
  zone    = "us-central1-a"       # You can change this to a zone closer to you
}

# Define the GCP Compute Engine instance
resource "google_compute_instance" "bhashasetu_vm" {
  name         = "bhashasetu-demo-vm"
  machine_type = "n1-standard-4" # 4 vCPUs, 15 GB RAM

  # Use a pre-built Deep Learning image with CUDA drivers installed
  boot_disk {
    initialize_params {
      image = "tf-latest-gpu-ubuntu-2004" # TensorFlow Enterprise with CUDA 11.3
    }
  }

  # Attach an NVIDIA T4 GPU
  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  # The instance needs a service account with access to run
  service_account {
    scopes = ["cloud-platform"]
  }

  # Allow HTTP traffic to our application
  network_interface {
    network = "default"
    access_config {
      // Ephemeral public IP
    }
  }

  # Add tags to apply firewall rules
  tags = ["http-server"]

  # This section ensures the GPU drivers are installed correctly on first boot
  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  # Execute our setup script on startup
  metadata_startup_script = file("startup-script.sh")
}

# Define a firewall rule to allow HTTP traffic on port 8000
resource "google_compute_firewall" "allow_http_8000" {
  name    = "allow-bhashasetu-http-8000"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8000"] # Port our Uvicorn server runs on
  }

  target_tags   = ["http-server"]
  source_ranges = ["0.0.0.0/0"] # Allow traffic from any IP
}

# Output the public IP address of the instance
output "instance_ip" {
  value = google_compute_instance.bhashasetu_vm.network_interface[0].access_config[0].nat_ip
}
