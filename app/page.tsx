"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Play,
  Square,
  Upload,
  Camera,
  AlertTriangle,
  Clock,
  Download,
  Eye,
  FileVideo,
  Info,
  CheckCircle,
} from "lucide-react"

interface DetectionLog {
  id: string
  timestamp: string
  duration: number
  frames: string[]
  confidence: number
  videoPath?: string
  source: string
  status: string
  type: string
}

export default function CrimeDetectionSystem() {
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectionLogs, setDetectionLogs] = useState<DetectionLog[]>([])
  const [currentAlert, setCurrentAlert] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [activeTab, setActiveTab] = useState("webcam")
  const [backendConnected, setBackendConnected] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const checkBackendConnection = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/health", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      })
      if (response.ok) {
        setBackendConnected(true)
        console.log("[v0] Backend connected successfully")
        return true
      } else {
        throw new Error("Backend not responding")
      }
    } catch (error) {
      console.log("[v0] Backend not available")
      setBackendConnected(false)
      return false
    }
  }

  const fetchDetectionLogs = async () => {
    const isConnected = await checkBackendConnection()

    if (isConnected) {
      try {
        const response = await fetch("http://localhost:5000/api/detections")
        if (response.ok) {
          const logs = await response.json()
          console.log("[v0] Fetched detection logs:", logs.length, "incidents")
          setDetectionLogs(logs)
        } else {
          throw new Error("Failed to fetch logs")
        }
      } catch (error) {
        console.log("[v0] Failed to fetch logs, using empty array")
        setDetectionLogs([])
      }
    } else {
      // Use mock data when backend is not available
      const mockDetectionLogs: DetectionLog[] = [
        {
          id: "demo-1",
          timestamp: "2024-01-15 14:30:25",
          duration: 4.2,
          frames: ["/security-camera-frame-1.png", "/security-camera-frame-2.png"],
          confidence: 0.87,
          videoPath: "/processed_video_1.mp4",
          source: "Upload",
          status: "active",
          type: "Violence",
        },
        {
          id: "demo-2",
          timestamp: "2024-01-15 12:15:10",
          duration: 3.8,
          frames: ["/security-camera-frame-3.png", "/security-camera-frame-4.png"],
          confidence: 0.92,
          videoPath: "/processed_video_2.mp4",
          source: "Upload",
          status: "active",
          type: "Violence",
        },
      ]
      setDetectionLogs(mockDetectionLogs)
    }
  }

  useEffect(() => {
    fetchDetectionLogs()
    const interval = setInterval(fetchDetectionLogs, 5000)
    return () => clearInterval(interval)
  }, [])

  const startWebcamDetection = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play()
      }
      setIsDetecting(true)

      if (backendConnected) {
        try {
          const response = await fetch("http://localhost:5000/api/start_detection", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ source: "webcam" }),
          })

          if (response.ok) {
            setCurrentAlert("Live detection started - Monitoring for violence...")
          } else {
            setCurrentAlert("Failed to start backend detection - Running in demo mode")
          }
        } catch (error) {
          setCurrentAlert("Backend not available - Running in demo mode")
        }
      } else {
        // Demo mode simulation
        setTimeout(() => {
          setCurrentAlert("Violence detected! Duration: 4.2s - Alerting authorities... (Demo Mode)")
          setTimeout(() => setCurrentAlert(null), 5000)
        }, 10000)
      }

      setTimeout(() => setCurrentAlert(null), 3000)
    } catch (error) {
      console.error("Error accessing webcam:", error)
      setCurrentAlert("Camera access denied. Please allow camera permissions.")
      setTimeout(() => setCurrentAlert(null), 5000)
    }
  }

  const stopDetection = async () => {
    setIsDetecting(false)
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }

    if (backendConnected) {
      try {
        await fetch("http://localhost:5000/api/stop_detection", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        })
        setCurrentAlert("Detection stopped")
        setTimeout(() => setCurrentAlert(null), 2000)
      } catch (error) {
        console.log("[v0] Failed to stop backend detection")
      }
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      if (videoRef.current) {
        videoRef.current.src = URL.createObjectURL(file)
      }
      console.log("[v0] File selected:", file.name)
    }
  }

  const processVideo = async () => {
    if (!selectedFile) return

    setIsProcessing(true)
    setCurrentAlert("Processing video... Please wait")

    if (backendConnected) {
      const formData = new FormData()
      formData.append("video", selectedFile)

      try {
        console.log("[v0] Uploading video to backend for processing")
        const response = await fetch("http://localhost:5000/api/process_video", {
          method: "POST",
          body: formData,
        })

        if (response.ok) {
          const result = await response.json()
          console.log("[v0] Processing result:", result)

          const incidentCount = result.incidents_detected || 0
          setCurrentAlert(`Processing complete! ${incidentCount} incidents detected.`)

          // Refresh logs after processing
          setTimeout(() => {
            fetchDetectionLogs()
          }, 1000)
        } else {
          const errorData = await response.json()
          setCurrentAlert(`Error processing video: ${errorData.error || "Unknown error"}`)
        }
      } catch (error) {
        console.error("[v0] Error processing video:", error)
        setCurrentAlert("Backend connection failed. Please check if the backend is running.")
      }
    } else {
      // Demo mode simulation
      setTimeout(() => {
        setCurrentAlert("Processing complete! 2 incidents detected. (Demo Mode)")
        // Add a new mock detection
        const newDetection: DetectionLog = {
          id: `demo-${Date.now()}`,
          timestamp: new Date().toLocaleString(),
          duration: 4.5,
          frames: ["/security-camera-frame-1.png", "/security-camera-frame-2.png"],
          confidence: 0.89,
          videoPath: "/demo_processed_video.mp4",
          source: "Upload",
          status: "active",
          type: "Violence",
        }
        setDetectionLogs((prev) => [newDetection, ...prev])
      }, 3000)
    }

    setIsProcessing(false)
    setTimeout(() => setCurrentAlert(null), 5000)
  }

  return (
    <div className="min-h-screen bg-white">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-black mb-2">SecureWatch</h1>
          <p className="text-gray-600 text-lg">Advanced Violence Detection & Monitoring System</p>
        </div>

        <Alert
          className={`mb-6 ${backendConnected ? "border-green-500 bg-green-50 text-green-800" : "border-blue-500 bg-blue-50 text-blue-800"}`}
        >
          {backendConnected ? <CheckCircle className="h-4 w-4" /> : <Info className="h-4 w-4" />}
          <AlertDescription>
            {backendConnected ? (
              <>
                <strong>Backend Connected:</strong> Full system operational with real-time violence detection and
                logging.
              </>
            ) : (
              <>
                <strong>Demo Mode:</strong> Backend not connected. To run the full system, start the Python Flask
                backend on localhost:5000. Currently showing mock data for demonstration.
              </>
            )}
          </AlertDescription>
        </Alert>

        {/* Alert Banner */}
        {currentAlert && (
          <Alert className="mb-6 border-red-500 bg-red-50 text-red-800">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="font-medium">{currentAlert}</AlertDescription>
          </Alert>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-gray-100 mb-6">
            <TabsTrigger value="webcam" className="text-black data-[state=active]:bg-white">
              Live Detection
            </TabsTrigger>
            <TabsTrigger value="upload" className="text-black data-[state=active]:bg-white">
              Video Analysis
            </TabsTrigger>
            <TabsTrigger value="logs" className="text-black data-[state=active]:bg-white">
              Detection Logs
            </TabsTrigger>
          </TabsList>

          <TabsContent value="webcam">
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Main Detection Panel */}
              <div className="lg:col-span-2">
                <Card className="bg-white border-gray-200 shadow-lg">
                  <CardHeader>
                    <CardTitle className="text-black flex items-center gap-2">
                      <Camera className="h-5 w-5" />
                      Live Webcam Detection
                      <Badge variant="outline" className="ml-2 text-orange-600 border-orange-600">
                        Under Work
                      </Badge>
                    </CardTitle>
                    <CardDescription className="text-gray-600">
                      Real-time violence detection with AI monitoring
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                      <video ref={videoRef} className="w-full h-full object-cover" muted />
                      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ display: "none" }} />
                      {!isDetecting && (
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="text-center text-gray-400">
                            <Camera className="h-16 w-16 mx-auto mb-4 opacity-50" />
                            <p>Click "Start Detection" to begin monitoring</p>
                          </div>
                        </div>
                      )}
                      {isDetecting && (
                        <div className="absolute top-4 left-4">
                          <Badge variant="destructive" className="animate-pulse">
                            <div className="w-2 h-2 bg-red-500 rounded-full mr-2 animate-ping" />
                            MONITORING
                          </Badge>
                        </div>
                      )}
                    </div>

                    <div className="flex gap-2 mt-4">
                      {!isDetecting ? (
                        <Button onClick={startWebcamDetection} className="bg-black hover:bg-gray-800 text-white">
                          <Play className="h-4 w-4 mr-2" />
                          Start Detection
                        </Button>
                      ) : (
                        <Button
                          onClick={stopDetection}
                          variant="outline"
                          className="border-gray-300 text-black bg-transparent"
                        >
                          <Square className="h-4 w-4 mr-2" />
                          Stop Detection
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* System Status Sidebar */}
              <div className="space-y-6">
                <Card className="bg-white border-gray-200 shadow-lg">
                  <CardHeader>
                    <CardTitle className="text-black text-lg">System Status</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Detection Engine</span>
                      <Badge className="bg-green-100 text-green-800 hover:bg-green-100">Active</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Backend Connection</span>
                      <Badge
                        className={
                          backendConnected
                            ? "bg-green-100 text-green-800 hover:bg-green-100"
                            : "bg-red-100 text-red-800 hover:bg-red-100"
                        }
                      >
                        {backendConnected ? "Connected" : "Demo Mode"}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Model Accuracy</span>
                      <span className="text-black font-medium">94.2%</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Total Incidents</span>
                      <span className="text-black font-medium">{detectionLogs.length}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="upload">
            <Card className="bg-white border-gray-200 shadow-lg">
              <CardHeader>
                <CardTitle className="text-black flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Video Analysis
                </CardTitle>
                <CardDescription className="text-gray-600">
                  Upload video files for violence detection analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                  <video ref={videoRef} className="w-full h-full object-cover" controls />
                  {!selectedFile && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center text-gray-400">
                        <Upload className="h-16 w-16 mx-auto mb-4 opacity-50" />
                        <p>Upload a video file to analyze</p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex gap-2 mt-4">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    variant="outline"
                    className="border-gray-300 text-black"
                  >
                    <Upload className="h-4 w-4 mr-2" />
                    Select Video
                  </Button>
                  {selectedFile && (
                    <Button
                      onClick={processVideo}
                      disabled={isProcessing}
                      className="bg-black hover:bg-gray-800 text-white"
                    >
                      {isProcessing ? "Processing..." : "Analyze Video"}
                    </Button>
                  )}
                </div>

                {selectedFile && <div className="text-sm text-gray-600 mt-2">Selected: {selectedFile.name}</div>}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="logs">
            <Card className="bg-white border-gray-200 shadow-lg">
              <CardHeader>
                <CardTitle className="text-black text-lg">Detection Logs & Evidence</CardTitle>
                <CardDescription className="text-gray-600">
                  Complete history of detected incidents with frames and processed videos
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {detectionLogs.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <AlertTriangle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No incidents detected yet</p>
                    <p className="text-sm mt-2">Upload a video or start live detection to see results here</p>
                  </div>
                ) : (
                  detectionLogs.map((log) => (
                    <div key={log.id} className="border border-gray-200 rounded-lg p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-sm text-gray-600">
                          <Clock className="h-4 w-4" />
                          {log.timestamp}
                        </div>
                        <Badge variant="destructive" className="text-xs">
                          {log.type || "Violence"} Detected
                        </Badge>
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Duration:</span>
                          <span className="ml-2 font-medium text-black">{log.duration}s</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Confidence:</span>
                          <span className="ml-2 font-medium text-black">{(log.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>

                      <div>
                        <h4 className="text-sm font-medium text-black mb-2">Key Frames Captured:</h4>
                        <div className="flex gap-2">
                          {log.frames && log.frames.length > 0 ? (
                            log.frames.slice(0, 4).map((frame, index) => (
                              <div key={index} className="relative">
                                <img
                                  src={
                                    backendConnected && frame.startsWith("http")
                                      ? frame
                                      : `/placeholder.svg?height=80&width=120&query=security camera frame ${index + 1}`
                                  }
                                  alt={`Frame ${index + 1}`}
                                  className="w-20 h-16 object-cover rounded border border-gray-300"
                                  onLoad={() => {
                                    console.log(`[v0] Successfully loaded frame: ${frame}`)
                                  }}
                                  onError={(e) => {
                                    console.log(`[v0] Failed to load frame: ${frame}`)
                                    console.log(`[v0] Backend connected: ${backendConnected}`)
                                    e.currentTarget.src = `/placeholder.svg?height=80&width=120&query=security camera frame ${index + 1}`
                                  }}
                                />
                                <div className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                                  {index + 1}
                                </div>
                                {!backendConnected && (
                                  <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded">
                                    <span className="text-white text-xs">Demo</span>
                                  </div>
                                )}
                              </div>
                            ))
                          ) : (
                            <div className="text-sm text-gray-500 flex items-center gap-2">
                              <AlertTriangle className="h-4 w-4" />
                              {backendConnected
                                ? "No frames captured for this incident"
                                : "Backend required to view frames"}
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="flex gap-2 pt-2 border-t border-gray-100">
                        <Button size="sm" variant="outline" className="border-gray-300 text-black bg-transparent">
                          <Eye className="h-3 w-3 mr-1" />
                          View Details
                        </Button>
                        {log.frames && log.frames.length > 0 && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="border-gray-300 text-black bg-transparent"
                            onClick={() => {
                              if (backendConnected && log.frames[0]) {
                                console.log(`[v0] Attempting to download frame: ${log.frames[0]}`)
                                window.open(log.frames[0], "_blank")
                              } else {
                                console.log(`[v0] Download blocked - Backend connected: ${backendConnected}`)
                                setCurrentAlert(
                                  "Backend required for frame download. Start the Python Flask backend on localhost:5000.",
                                )
                                setTimeout(() => setCurrentAlert(null), 4000)
                              }
                            }}
                          >
                            <Download className="h-3 w-3 mr-1" />
                            Download Frames
                          </Button>
                        )}
                        {log.videoPath && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="border-gray-300 text-black bg-transparent"
                            onClick={() => {
                              if (backendConnected) {
                                window.open(`http://localhost:5000/api/download_video/${log.id}`, "_blank")
                              } else {
                                setCurrentAlert("Backend required for video download. Currently in demo mode.")
                                setTimeout(() => setCurrentAlert(null), 3000)
                              }
                            }}
                          >
                            <FileVideo className="h-3 w-3 mr-1" />
                            View Video
                          </Button>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
