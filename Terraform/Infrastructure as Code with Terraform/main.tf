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
  #  Below for 'Changing resources' section and after
  #tags = ["web", "dev"]
  # Below for 'Defining a provisioner' section
  #provisioner "local-exec" {
  #  command = "echo ${google_compute_instance.vm_instance.name}: ${google_compute_instance.vm_instance.network_interface[0].access_config[0].nat_ip} >> ip_address.txt"
  #}
  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      # Below for 'Destructive changes' section
      #image = "cos-cloud/cos-stable"
    }
  }
  network_interface {
    network = google_compute_network.vpc_network.name
    access_config {
    }
    # Modified for the 'Assigning a static IP address' section below
    #network = google_compute_network.vpc_network.self_link
    #access_config {
    # nat_ip = google_compute_address.vm_static_ip.address
    #}
  }
  # Static IP added below
  resource "google_compute_address" "vm_static_ip" {
    name = "terraform-static-ip"
  }
}
# New resource for the storage bucket our application will use.
resource "google_storage_bucket" "example_bucket" {
  name = "<UNIQUE-BUCKET-NAME>"
  location = "US"
  website {
    main_page_suffix = "index.html"
    not_found_page = "404.html"
  }
}
# Create a new instance that uses the bucket
resource "google_compute_instance" "another_instance" {
  # Tells Terraform that this VM instance must be created only after the
  # storage bucket has been created.
  depends_on = [google_storage_bucket.example_bucket]
  name = "terraform-instance-2"
  machine_type = "f1-micro"
  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
    }
  }
  network_interface {
    network = google_compute_network.vpc_network.self_link
    access_config {
    }
  }
}
    
