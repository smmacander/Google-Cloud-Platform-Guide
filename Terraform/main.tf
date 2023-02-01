terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}
provider "google" {
  version = "3.5.0"
  project = "<PROJECT_ID>"
  region = "us-central1"
  zone = "us-central1-c"
}
resource "google_compute_network" "vpc_network" {
  name = "terraform-network"
}

# Compute Instance added below

resource "google_compute_instance" "vm_instance" {
  name = "terraform-instance"
  machine_type = "f1-micro"
  #  Below for 'Changing resources' section
  #tags = ["web", "dev"]
  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      # Below for 'Destructive changes' section
      #image = "cos-cloud/cos-stable"
    }
  }
  network_interface {
    network = google_compute_network.vpc_network.name
    access_config{
    }
  }
  # Static IP added below
  resource "google_compute_address" "vm_static_ip" {
    name = "terraform-static-ip"
  }
}
