# Sources and attribution

Official OmniHand 2025 left/right URDF archives, AgiBot, released 2026-08-27.
See provenance.json for original URLs and archive SHA-256; file_hashes.json records extracted originals.
Each original package.xml declares BSD. Original package files and attribution are retained unchanged.

kinematics.json records gesture outputs and coupling samples from AgiBot OmniHand SDK 1.1.10,
revision 1b552ea77c2d61adef327d179dc9f6b47405c628, licensed under Mulan PSL v2.
The SDK binaries are not needed at runtime. The upstream README declares Mulan PSL v2 but this revision has no standalone LICENSE file.
See the repository source:
https://github.com/AgibotTech/agillink_omnihand_sdk

The current simulation uses the URDF mimic relationships; captured SDK coupling coefficients
are evidence of the source version, not the simulation's coupling model.
