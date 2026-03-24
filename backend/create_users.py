import json
import urllib.request
import urllib.error

API_URL = "http://localhost:8000/auth/register"

def create_user(name, email, role, password):
    payload = json.dumps({
        "name": name,
        "email": email,
        "role": role,
        "password": password
    }).encode('utf-8')
    
    req = urllib.request.Request(API_URL, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 201:
                print(f"✅ Success: Created {role} -> {email}")
    except urllib.error.HTTPError as e:
        if e.code == 409:
            print(f"⚠️ User {email} already exists!")
        elif e.code == 401:
            print(f"❌ Vercel Protection is blocking the API! Please turn off 'Vercel Authentication' in your Vercel Settings.")
        else:
            print(f"❌ Error creating {email}: HTTP {e.code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Creating default MindMesh users in the Live Cloud Database...\n")
    
    create_user("System Admin", "admin@mindmesh.edu", "admin", "Admin123!")
    create_user("Dr. Sarah Jenkins", "counselor@mindmesh.edu", "counselor", "Help2024!")
    create_user("Alex Freshman", "student@mindmesh.edu", "student", "Student123!")
    
    print("\n🎉 Done! Try logging in with these credentials.")
