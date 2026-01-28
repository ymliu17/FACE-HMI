═══════════════════════════════════════════════════════════════
    BrainWellness Complete Suite - Version 1.0.0
    MULTI-ARCHITECTURE BUILD
    Works on BOTH Intel/AMD AND Apple Silicon (M1/M2/M3)!
    Includes: API, Games WebApp & Results Viewer
    Build Date: 20251027_123841
═══════════════════════════════════════════════════════════════

✨ AUTOMATIC ARCHITECTURE SELECTION ✨
The images automatically use the correct architecture for your system:
- Intel/AMD (x86_64/amd64) - Windows PCs, Intel Macs, most laptops
- Apple Silicon (arm64) - M1/M2/M3 Macs, some newer Windows devices

You don't need to do anything special - Docker handles it automatically!

═══════════════════════════════════════════════════════════════
OPTIONAL: Check Your Architecture
═══════════════════════════════════════════════════════════════

Mac/Linux:
  ./check-architecture.sh

Windows (PowerShell):
  .\check-architecture.ps1

This is optional - images work on both architectures automatically!

═══════════════════════════════════════════════════════════════
STEP 1: Install Docker Desktop
═══════════════════════════════════════════════════════════════

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
   - Automatically detects your system and downloads the right version
   - For M1/M2/M3 Macs: Downloads "Apple Chip" version
   - For Intel Macs/Windows: Downloads standard version

2. Install and restart your computer

3. Open Docker Desktop and wait for it to start (whale icon should be visible)

═══════════════════════════════════════════════════════════════
STEP 2: Import Images and Start
═══════════════════════════════════════════════════════════════

Open Terminal (Mac) or PowerShell (Windows) and navigate to this folder:

Mac/Linux:
----------
cd /path/to/brainwellness-distribution-multiarch
docker load < brainwellness-api-image.tar.gz
docker load < brainwellness-games-webapp-image.tar.gz
docker load < brainwellness-resultsviewer-image.tar.gz
docker load < database-init-image.tar.gz
docker-compose up -d

Windows (PowerShell):
--------------------
cd C:\path\to\brainwellness-distribution-multiarch
docker load -i brainwellness-api-image.tar.gz
docker load -i brainwellness-games-webapp-image.tar.gz
docker load -i brainwellness-resultsviewer-image.tar.gz
docker load -i database-init-image.tar.gz
docker-compose up -d

🎉 The images will automatically use the correct architecture!
🎉 The database will restore AUTOMATICALLY on first start!

═══════════════════════════════════════════════════════════════
STEP 3: Verify in Docker Desktop (Optional)
═══════════════════════════════════════════════════════════════

1. Open Docker Desktop
2. Go to "Containers" tab
3. You should see 5 containers:
   - database-init (will complete and exit)
   - sqlserver (running)
   - brainwellness-api (running)
   - brainwellness-games-webapp (running)
   - brainwellness-resultsviewer (running)
4. Click on "database-init" → Logs to see "Database restored successfully!"

═══════════════════════════════════════════════════════════════
STEP 4: Access the Applications
═══════════════════════════════════════════════════════════════

Wait about 30 seconds for everything to start, then open your web browser:
- http://localhost:8080/swagger  (API documentation & testing)
- http://localhost:8082          (Games Web Application)
- http://localhost:8084          (Results Viewer)

You should see all three applications running!

═══════════════════════════════════════════════════════════════
MANAGING THE APPLICATION
═══════════════════════════════════════════════════════════════

Using Command Line (Recommended):
---------------------------------
START:    docker-compose up -d
STOP:     docker-compose down
RESTART:  docker-compose restart
LOGS:     docker-compose logs -f

Using Docker Desktop UI:
-----------------------
START:
1. Open Docker Desktop
2. Go to "Containers" tab
3. Find the container group
4. Click the Play ▶ button

STOP:
1. Go to "Containers" tab
2. Click the Stop ⏹ button (or trash icon to remove)

VIEW LOGS:
1. Click on a container name
2. Click "Logs" tab
3. Look for "Database restored successfully!" message

═══════════════════════════════════════════════════════════════
ARCHITECTURE COMPATIBILITY
═══════════════════════════════════════════════════════════════

✅ Intel/AMD Macs (x86_64)
✅ Apple Silicon Macs (M1/M2/M3/M4)
✅ Windows with Intel/AMD processors
✅ Windows with ARM processors (newer Surface devices)
✅ Linux with Intel/AMD processors
✅ Linux with ARM processors

The images contain BOTH architectures and Docker automatically
selects the correct one for your system!

═══════════════════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════

Problem: "Port already in use"
Solution: Another app is using port 8080. Close other apps or ask for help.

Problem: Can't access localhost:8080
Solution: 
- Make sure containers show "Running" (green) in Docker Desktop
- Wait 1-2 minutes for the API to fully start
- Check Windows Firewall isn't blocking the port

Problem: Containers keep restarting
Solution:
- Wait 1-2 minutes for database to initialize
- Check logs for "Database restored successfully!"
- If still failing, share the logs with your team

Problem: Database seems empty
Solution:
- Check that you imported ALL 4 images (api + games + results + database-init)
- Look in logs for database-init container
- Should see "Database restored successfully!"

Problem: Can't access games or results viewer
Solution:
- Make sure you loaded all image files
- Check that all containers are running in Docker Desktop
- Verify ports 8080, 8082, and 8084 are not in use by other apps

Problem: "exec format error" or architecture mismatch
Solution:
- This should NOT happen with multi-arch images
- If you see this, the image may not have been built correctly
- Run the check-architecture script to verify your system
- Contact your team lead for a rebuild

═══════════════════════════════════════════════════════════════
DATABASE ACCESS (Optional - Advanced Users)
═══════════════════════════════════════════════════════════════

If you need to access the SQL Server database directly:
- Server: localhost,1433
- Username: sa
- Password: URHealthLab2025!
- Database: BrainWellness

Use Azure Data Studio, SQL Server Management Studio, or similar tools.

═══════════════════════════════════════════════════════════════
NEED HELP?
═══════════════════════════════════════════════════════════════

1. Check your distribution version:
   - Open VERSION.txt file
   - Or check the top of this README
   - Or check docker-compose.yml header

2. Run the architecture check script (optional):
   Mac/Linux: ./check-architecture.sh
   Windows: .\check-architecture.ps1

3. Take a screenshot of any error in Docker Desktop

4. Check the logs (click container → Logs tab)

5. Contact your team lead or IT support

6. Include:
   - Distribution version (from VERSION.txt)
   - Your system type (Mac/Windows, Intel/Apple Silicon)
   - What you were doing when the error occurred
   - Screenshot of error and logs

═══════════════════════════════════════════════════════════════
VERSION INFORMATION
═══════════════════════════════════════════════════════════════

Distribution Version: 1.0.0
Build Date: 20251027_123841

To check which version you have, look at the top of this README or
the docker-compose.yml file header.

═══════════════════════════════════════════════════════════════
That's it! Works on any architecture - completely automatic! 🚀
═══════════════════════════════════════════════════════════════
