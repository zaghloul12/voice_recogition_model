# quick_start.py
import os
import sys
import argparse
import time

def check_faces_directory(faces_dir):
    """Check if the faces directory exists and has content"""
    if not os.path.exists(faces_dir):
        print(f"❌ Directory '{faces_dir}' does not exist!")
        print(f"📁 Please create the directory with subfolders for each person.")
        print(f"   Example: {faces_dir}/John/photo1.jpg")
        return False
    
    # Check if there are person subfolders
    person_folders = [f for f in os.listdir(faces_dir) 
                     if os.path.isdir(os.path.join(faces_dir, f))]
    
    if not person_folders:
        print(f"❌ No person subfolders found in '{faces_dir}'")
        print(f"📁 Please create subfolders for each person.")
        print(f"   Example: {faces_dir}/John/photo1.jpg")
        return False
    
    print(f"✅ Found {len(person_folders)} people in '{faces_dir}':")
    for person in person_folders:
        person_dir = os.path.join(faces_dir, person)
        images = [f for f in os.listdir(person_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        print(f"   👤 {person}: {len(images)} photos")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Quick start for Robot Assistant')
    parser.add_argument('--faces-dir', type=str, default='known_faces',
                       help='Path to folder with person subfolders')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID to use')
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("🤖 Robot Assistant - Quick Start")
    print("="*50)
    
    # Check faces directory
    print(f"\n📁 Checking faces directory: {args.faces_dir}")
    if not check_faces_directory(args.faces_dir):
        print("\n💡 Example structure:")
        print(f"   {args.faces_dir}/")
        print(f"   ├── John/")
        print(f"   │   ├── john_1.jpg")
        print(f"   │   └── john_2.jpg")
        print(f"   ├── Sarah/")
        print(f"   │   └── sarah_1.jpg")
        print(f"   └── Michael/")
        print(f"       └── michael_1.jpg")
        print("\n❌ Please set up your faces directory first.")
        sys.exit(1)
    
    # Check if model exists, if not it will be trained
    if os.path.exists("face_model.pkl"):
        print("✅ Model file exists, loading...")
    else:
        print("🔄 No model found, will train from your photos...")
    
    # Run the robot
    print("\n🚀 Starting Robot Assistant...")
    print("📝 Voice commands: say 'what' to wake me up")
    print("📸 Face recognition: I'll greet recognized faces")
    print("\n⏳ Loading...")
    
    # Import and run main
    import main
    sys.argv = ['main.py', '--camera', str(args.camera), '--faces-dir', args.faces_dir]
    main.main()

if __name__ == "__main__":
    main()