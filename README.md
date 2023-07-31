Progress (as of 2023.07.31)
- Implementation of communication module between server/client (completed, but planned to use JayB module later)
- Apply encoding/decoding module using avatarify (Server and client must be run on different PCs, and cuda environment must be set for each)
- Transmit and receive key frames and avatarify feature points and display them on the screen
- Completion of screen composition using pyqt5
- Application of JayB Prototype module completed (Implementation completed to apply the previously received content as a parameter classification)

Future progress (as of 2023.07.31)
- Screen output part after video encoding/decoding process is in progress (to be completed ~2023/8/22)
- Audio module processing (scheduled to implement only sound output function until 8/22, synchronization with video, etc. will be processed later)
- Payload module processing of transmitted and received packets (previously implemented, then rolled back, this part will be applied from 2023/08/22 until the end of the year)
- Application of SNNM mode (Details of application period and contents will be confirmed after additional review)
- Applied when JayB SDK module is completed (scheduled to be applied by the end of 2023)

Build Configuration
  * Environment variable
    - PYTHONUNBUFFERED=1;PYTHONPATH=%PYTHONPATH%\\\;\;[Working Directory]\\;[Working Directory]\fomm

Execute parameter
  * Server  
    --config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar --is-server --listen-port [LISTEN_PORT] --keyframe-period 11000
    
  * Client  
    --config fomm/config/vox-adv-256.yaml --relative --adapt_scale --no-pad --checkpoint vox-adv-cpk.pth.tar --server-ip [SERVER_IP] --server-port [SERVER_PORT] --keyframe-period 11000

