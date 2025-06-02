"use client";

import { motion } from "framer-motion";
import { Check, CircleStop, Pencil, Folder } from "lucide-react";

import Action from "@/components/action";
import Markdown from "@/components/markdown";
import QuestionInput from "@/components/question-input";
import { ActionStep, Message } from "@/typings/agent";
import { getFileIconAndColor } from "@/utils/file-utils";
import { Button } from "./ui/button";
import EditQuestion from "./edit-question";

interface ChatMessageProps {
  messages: Message[];
  isLoading: boolean;
  isCompleted: boolean;
  isStopped: boolean;
  workspaceInfo: string;
  handleClickAction: (
    data: ActionStep | undefined,
    showTabOnly?: boolean
  ) => void;
  isUploading: boolean;
  isReplayMode: boolean;
  currentQuestion: string;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  setCurrentQuestion: (value: string) => void;
  handleKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  handleQuestionSubmit: (question: string) => void;
  handleFileUpload: (
    e: React.ChangeEvent<HTMLInputElement>,
    dontAddToUserMessage?: boolean
  ) => void;
  handleGoogleDriveAuth: () => Promise<boolean>;
  isGoogleDriveConnected?: boolean;
  isGeneratingPrompt: boolean;
  handleEnhancePrompt: () => void;
  handleCancel: () => void;
  editingMessage?: Message;
  setEditingMessage: (message?: Message) => void;
  handleEditMessage: (newQuestion: string) => void;
  googlePickerApiLoaded: boolean;
  setIsGoogleDriveConnected: (value: boolean) => void;
}

const ChatMessage = ({
  messages,
  isLoading,
  isCompleted,
  isStopped,
  workspaceInfo,
  isUploading,
  currentQuestion,
  messagesEndRef,
  isReplayMode,
  handleClickAction,
  setCurrentQuestion,
  handleKeyDown,
  handleQuestionSubmit,
  handleFileUpload,
  isGeneratingPrompt,
  handleEnhancePrompt,
  handleCancel,
  editingMessage,
  setEditingMessage,
  handleEditMessage,
  handleGoogleDriveAuth,
  isGoogleDriveConnected,
  googlePickerApiLoaded,
  setIsGoogleDriveConnected,
}: ChatMessageProps) => {
  // Helper function to check if a message is the latest user message
  const isLatestUserMessage = (
    message: Message,
    allMessages: Message[]
  ): boolean => {
    const userMessages = allMessages.filter((msg) => msg.role === "user");
    return (
      userMessages.length > 0 &&
      userMessages[userMessages.length - 1].id === message.id
    );
  };

  return (
    <div className="col-span-4">
      <motion.div
        className="p-4 pt-0 w-full h-full max-h-[calc(100vh-230px)] overflow-y-auto relative"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.3 }}
      >
        {messages.map((message, index) => (
          <motion.div
            key={message.id}
            className={`mb-4 ${
              message.role === "user" ? "text-right" : "text-left"
            } ${message.role && !message.files && "mb-8"}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * index, duration: 0.3 }}
          >
            {message.files && message.files.length > 0 && (
              <div className="flex flex-col gap-2 mb-2">
                {(() => {
                  // First, identify any folders in the files array
                  const folderFiles = message.files.filter((fileName) =>
                    fileName.match(/^folder:(.+):(\d+)$/)
                  );

                  // Extract folder names for filtering
                  const folderNames = folderFiles
                    .map((folderFile) => {
                      const match = folderFile.match(/^folder:(.+):(\d+)$/);
                      return match ? match[1] : null;
                    })
                    .filter(Boolean) as string[];

                  // Create a list of files to display:
                  // 1. Include all folder entries
                  // 2. Only include individual files that are NOT part of any folder
                  const filesToDisplay = message.files.filter((fileName) => {
                    // If it's a folder entry, always include it
                    if (fileName.match(/^folder:(.+):(\d+)$/)) {
                      return true;
                    }

                    // For regular files, exclude them if they might be part of a folder
                    // This is a simple heuristic that checks if the filename contains any folder name
                    for (const folderName of folderNames) {
                      // If the file appears to be from a Google Drive folder, exclude it
                      if (fileName.includes(folderName)) {
                        return false;
                      }
                    }

                    // Include all other files (they're not part of folders)
                    return true;
                  });

                  return filesToDisplay.map((fileName, fileIndex) => {
                    // Check if the file is a folder
                    const isFolderMatch = fileName.match(/^folder:(.+):(\d+)$/);
                    if (isFolderMatch) {
                      const folderName = isFolderMatch[1];
                      const fileCount = parseInt(isFolderMatch[2], 10);

                      return (
                        <div
                          key={`${message.id}-folder-${fileIndex}`}
                          className="inline-block ml-auto bg-[#35363a] text-white rounded-2xl px-4 py-3 border border-gray-700 shadow-sm"
                        >
                          <div className="flex items-center gap-3">
                            <div className="flex items-center justify-center w-12 h-12 bg-blue-600 rounded-xl">
                              <Folder className="size-6 text-white" />
                            </div>
                            <div className="flex flex-col">
                              <span className="text-base font-medium">
                                {folderName}
                              </span>
                              <span className="text-left text-sm text-gray-500">
                                {fileCount} {fileCount === 1 ? "file" : "files"}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    }

                    // Handle regular files as before
                    // Check if the file is an image
                    const isImage =
                      fileName.match(
                        /\.(jpeg|jpg|gif|png|webp|svg|heic|bmp)$/i
                      ) !== null;

                    if (
                      isImage &&
                      message.fileContents &&
                      message.fileContents[fileName]
                    ) {
                      return (
                        <div
                          key={`${message.id}-file-${fileIndex}`}
                          className="inline-block ml-auto rounded-3xl overflow-hidden max-w-[320px]"
                        >
                          <div className="w-40 h-40 rounded-xl overflow-hidden">
                            <img
                              src={message.fileContents[fileName]}
                              alt={fileName}
                              className="w-full h-full object-cover"
                            />
                          </div>
                        </div>
                      );
                    }

                    // For non-image files, use the existing code
                    const { IconComponent, bgColor, label } =
                      getFileIconAndColor(fileName);

                    return (
                      <div
                        key={`${message.id}-file-${fileIndex}`}
                        className="inline-block ml-auto bg-[#35363a] text-white rounded-2xl px-4 py-3 border border-gray-700 shadow-sm"
                      >
                        <div className="flex items-center gap-3">
                          <div
                            className={`flex items-center justify-center w-12 h-12 ${bgColor} rounded-xl`}
                          >
                            <IconComponent className="size-6 text-white" />
                          </div>
                          <div className="flex flex-col">
                            <span className="text-base font-medium">
                              {fileName}
                            </span>
                            <span className="text-left text-sm text-gray-500">
                              {label}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  });
                })()}
              </div>
            )}

            {message.content && (
              <motion.div
                className={`inline-block text-left rounded-lg ${
                  message.role === "user"
                    ? "bg-[#35363a] p-3 max-w-[80%] text-white border border-[#3A3B3F] shadow-sm whitespace-pre-wrap"
                    : "text-white"
                } ${
                  editingMessage?.id === message.id ? "w-full max-w-none" : ""
                }`}
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                transition={{
                  type: "spring",
                  stiffness: 500,
                  damping: 30,
                }}
              >
                {message.role === "user" ? (
                  <div>
                    {editingMessage?.id === message.id ? (
                      <EditQuestion
                        editingMessage={message.content}
                        handleCancel={() => setEditingMessage(undefined)}
                        handleEditMessage={handleEditMessage}
                      />
                    ) : (
                      <div className="relative group">
                        <div className="text-left">{message.content}</div>
                        {isLatestUserMessage(message, messages) &&
                          !isReplayMode && (
                            <div className="absolute -bottom-[45px] -right-[20px] opacity-0 group-hover:opacity-100 transition-opacity">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="text-xs cursor-pointer hover:!bg-transparent"
                                onClick={() => {
                                  setEditingMessage(message);
                                }}
                              >
                                <Pencil className="size-3 mr-1" />
                              </Button>
                            </div>
                          )}
                      </div>
                    )}
                  </div>
                ) : (
                  <Markdown>{message.content}</Markdown>
                )}
              </motion.div>
            )}

            {message.action && (
              <motion.div
                className="mt-2"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * index, duration: 0.3 }}
              >
                <Action
                  workspaceInfo={workspaceInfo}
                  type={message.action.type}
                  value={message.action.data}
                  onClick={() => handleClickAction(message.action, true)}
                />
              </motion.div>
            )}
          </motion.div>
        ))}

        {isLoading && (
          <motion.div
            className="mb-4 text-left"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              type: "spring",
              stiffness: 300,
              damping: 30,
            }}
          >
            <motion.div
              className="inline-block p-3 text-left rounded-lg bg-neutral-800/90 text-white backdrop-blur-sm"
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              transition={{
                type: "spring",
                stiffness: 400,
                damping: 25,
              }}
            >
              <div className="flex items-center gap-3">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 bg-white rounded-full animate-[dot-bounce_1.2s_ease-in-out_infinite_0ms]" />
                  <div className="w-2 h-2 bg-white rounded-full animate-[dot-bounce_1.2s_ease-in-out_infinite_200ms]" />
                  <div className="w-2 h-2 bg-white rounded-full animate-[dot-bounce_1.2s_ease-in-out_infinite_400ms]" />
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}

        {isCompleted && (
          <div className="flex gap-x-2 items-center bg-[#25BA3B1E] text-green-600 text-sm p-2 rounded-full">
            <Check className="size-4" />
            <span>II-Agent has completed the current task.</span>
          </div>
        )}

        {isStopped && (
          <div className="flex gap-x-2 items-center bg-[#ffbf361f] text-yellow-300 text-sm p-2 rounded-full">
            <CircleStop className="size-4" />
            <span>II-Agent has stopped, send a new message to continue.</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </motion.div>
      <motion.div
        className="sticky bottom-0 left-0 w-full"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.3 }}
      >
        <QuestionInput
          className="p-4 pb-0 w-full max-w-none"
          textareaClassName="h-30 w-full"
          placeholder="Ask me anything..."
          value={currentQuestion}
          setValue={setCurrentQuestion}
          handleKeyDown={handleKeyDown}
          handleSubmit={handleQuestionSubmit}
          handleFileUpload={handleFileUpload}
          handleGoogleDriveAuth={handleGoogleDriveAuth}
          isGoogleDriveConnected={isGoogleDriveConnected}
          isUploading={isUploading}
          isGeneratingPrompt={isGeneratingPrompt}
          handleEnhancePrompt={handleEnhancePrompt}
          isLoading={isLoading}
          handleCancel={handleCancel}
          googlePickerApiLoaded={googlePickerApiLoaded}
          setIsGoogleDriveConnected={setIsGoogleDriveConnected}
        />
      </motion.div>
    </div>
  );
};

export default ChatMessage;
