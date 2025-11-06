"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Play, Square, Upload, Camera, AlertTriangle, Clock, CheckCircle, Info, RefreshCw } from "lucide-react"

interface DetectionLog {
  id: string
  timestamp: string
  confidence: number
  frames: string[]
  source: string
  type: string
  severity: string
  status: string
}

export default function CrimeDetectionSystem() {
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectionLogs, setDetectionLogs] = useState<DetectionLog[]>([])
  const [currentAlert, setCurrentAlert] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [activeTab, setActiveTab] = useState("webcam")
  const [backendConnected, setBackendConnected] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const checkBackendConnection = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/health", {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      })
      setBackendConnected(response.ok)
      return response.ok
    } catch (error) {
      console.log("[v0] Backend connection failed")
      setBackendConnected(false)
      return false
    }
  }

  const fetchDetectionLogs = async () => {
    const isConnected = await checkBackendConnection()

    if (isConnected) {
      try {
        setIsRefreshing(true)
        const response = await fetch("http://localhost:5000/api/detections")
        if (response.ok) {
          const logs = await response.json()
          console.log("[v0] Fetched detection logs:", logs.length, "incidents")
          setDetectionLogs(logs)
        }
      } catch (error) {
        console.log("[v0] Failed to fetch logs")
      } finally {
        setIsRefreshing(false)
      }
    }
  }

  useEffect(() => {
    fetchDetectionLogs()
    const interval = setInterval(fetchDetectionLogs, 2000)
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
        const response = await fetch("http://localhost:5000/api/start_detection", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: "webcam" }),
        })
        if (response.ok) {
          setCurrentAlert("Live detection started - Monitoring for crime...")
        }
      }
    } catch (error) {
      setCurrentAlert("Camera access denied.")
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
          headers: { "Content-Type": "application/json" },
        })
        setCurrentAlert("Detection stopped")
        setTimeout(() => setCurrentAlert(null), 2000)
      } catch (error) {
        console.log("[v0] Failed to stop detection")
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
    }
  }

  const processVideo = async () => {
    if (!selectedFile) return

    setIsProcessing(true)
    setCurrentAlert("Processing video... Detecting crimes frame by frame")

    if (backendConnected) {
      const formData = new FormData()
      formData.append("video", selectedFile)

      try {
        const response = await fetch("http://localhost:5000/api/process_video", {
          method: "POST",
          body: formData,
        })

        if (response.ok) {
          const result = await response.json()
          const incidentCount = result.incidents_detected || 0
          setCurrentAlert(
            `Processing complete! ${incidentCount} crime incidents detected with alerts sent to Telegram.`,
          )

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
        setCurrentAlert("Backend connection failed.")
      }
    }

    setIsProcessing(false)
    setTimeout(() => setCurrentAlert(null), 5000)
  }

  return (
    <div className="min-h-screen bg-white">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-black mb-2">SecureWatch</h1>
          <p className="text-gray-600 text-lg">Real-Time Violence Detection & Alert System</p>
        </div>

        <Alert
          className={`mb-6 ${
            backendConnected
              ? "border-green-500 bg-green-50 text-green-800"
              : "border-blue-500 bg-blue-50 text-blue-800"
          }`}
        >
          {backendConnected ? <CheckCircle className="h-4 w-4" /> : <Info className="h-4 w-4" />}
          <AlertDescription>
            {backendConnected ? (
              <>
                <strong>Backend Connected:</strong> Real-time detection active. All crimes detected will be sent to
                Telegram with 3-frame evidence.
              </>
            ) : (
              <>
                <strong>Demo Mode:</strong> Backend not connected. Start the Python Flask backend on localhost:5000.
              </>
            )}
          </AlertDescription>
        </Alert>

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
              <div className="lg:col-span-2">
                <Card className="bg-white border-gray-200 shadow-lg">
                  <CardHeader>
                    <CardTitle className="text-black flex items-center gap-2">
                      <Camera className="h-5 w-5" />
                      Live Webcam Detection
                      <Badge variant="outline" className="ml-2 text-green-600 border-green-600">
                        Real-Time
                      </Badge>
                    </CardTitle>
                    <CardDescription className="text-gray-600">
                      Real-time violence detection with instant Telegram alerts
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                      <video ref={videoRef} className="w-full h-full object-cover" muted />
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
                            LIVE MONITORING
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

              <div className="space-y-6">
                <Card className="bg-white border-gray-200 shadow-lg">
                  <CardHeader>
                    <CardTitle className="text-black text-lg flex items-center justify-between">
                      System Status
                      {isRefreshing && <RefreshCw className="h-4 w-4 animate-spin" />}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Detection Engine</span>
                      <Badge className="bg-green-100 text-green-800">Active</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Backend Status</span>
                      <Badge className={backendConnected ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}>
                        {backendConnected ? "Connected" : "Offline"}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Total Incidents</span>
                      <span className="text-black font-medium">{detectionLogs.length}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Frames per Incident</span>
                      <span className="text-black font-medium">Max 3</span>
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
                  Upload video files for violence detection and instant Telegram alerts
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
                  Complete history of detected incidents with evidence frames sent to Telegram
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {detectionLogs.length === 0 ? (
                  <div className="text-center py-8 text-gray-500">
                    <AlertTriangle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No crimes detected yet</p>
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
                          {log.type} - {log.severity}
                        </Badge>
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Confidence:</span>
                          <span className="ml-2 font-medium text-black">{(log.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Frames:</span>
                          <span className="ml-2 font-medium text-black">{log.frames.length}/3</span>
                        </div>
                      </div>

                      <div>
                        <h4 className="text-sm font-medium text-black mb-2">Evidence Frames (Sent to Telegram):</h4>
                        <div className="flex gap-2">
                          {log.frames && log.frames.length > 0 ? (
                            log.frames.map((frame, index) => (
                              <div key={index} className="relative">
                                <img
                                  src={frame || "/placeholder.svg"}
                                  alt={`Frame ${index + 1}`}
                                  className="w-20 h-16 object-cover rounded border border-gray-300"
                                  onError={(e) => {
                                    e.currentTarget.src = "/crime-detection-frame.jpg"
                                  }}
                                />
                                <div className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center font-bold">
                                  {index + 1}
                                </div>
                              </div>
                            ))
                          ) : (
                            <div className="text-sm text-gray-500">No frames available</div>
                          )}
                        </div>
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
